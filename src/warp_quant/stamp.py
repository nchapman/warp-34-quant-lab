"""Artifact stamping for our quants: serving generation_config + chat template."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from warp_quant.cachepath import resolve_snapshot
from warp_quant.models import serving_params

RELEASE_TEMPLATE = Path(__file__).parent / "release_chat_template.jinja"
RELEASE_TEMPLATE_SHA = "e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2"

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "do_sample")

PROCESSOR_CONFIG_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)


def _vision_config(snapshot_dir: str | Path, what: str) -> dict[str, Any] | None:
    p = Path(snapshot_dir) / "config.json"
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(
            f"{what}: config.json is not parseable JSON ({e}) — refusing to gate a processor fill on it"
        ) from e
    if not isinstance(cfg, dict):
        raise SystemExit(
            f"{what}: config.json is not a JSON object ({type(cfg).__name__}) — refusing to gate a processor fill on it"
        )
    v = cfg.get("vision_config")
    if v is not None and not isinstance(v, dict):
        raise SystemExit(
            f"{what}: vision_config is not a JSON object ({type(v).__name__}) — refusing to gate a processor fill on it"
        )
    return v


def ensure_processor_configs(
    snapshot_dir: str | Path,
    fallback: str | Path | None = None,
    fallback_revision: str | None = None,
    dry_run: bool = False,
    offline: bool | None = None,
) -> list[str]:
    if not fallback:
        return []
    d = Path(snapshot_dir)
    if all((d / n).exists() for n in PROCESSOR_CONFIG_FILES):
        return []
    src = Path(fallback)
    if not src.exists():
        want_offline = offline if offline is not None else os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        try:
            src = Path(resolve_snapshot(str(fallback), revision=fallback_revision, offline=want_offline))
        except Exception as e:
            raise SystemExit(
                f"processor_hub_fallback {fallback!r} cannot be resolved ({type(e).__name__}: {e}) — "
                "snapshot_download it online first"
            ) from e
    a = _vision_config(d, "target snapshot")
    b = _vision_config(src, f"fallback {fallback!r}")
    todo = [n for n in PROCESSOR_CONFIG_FILES if not (d / n).exists() and (src / n).exists()]
    if not todo:
        return []
    if a is None or b is None:
        raise SystemExit(
            f"processor_hub_fallback {fallback!r}: vision_config missing on one side — refusing ungated processor fill"
        )
    if {k: x for k, x in a.items() if k != "dtype"} != {k: x for k, x in b.items() if k != "dtype"}:
        raise SystemExit(f"processor_hub_fallback {fallback!r} vision tower differs from target snapshot")
    added: list[str] = []
    for name in todo:
        s = src / name
        try:
            parsed = json.loads(s.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise SystemExit(
                f"processor_hub_fallback {fallback!r}: {name} is not parseable UTF-8 JSON ({e})"
            ) from e
        if not dry_run:
            fd, tmp = tempfile.mkstemp(dir=d, prefix=name + ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps(parsed, indent=2))
                os.replace(tmp, d / name)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        added.append(name)
    return added


def ensure_license_files(
    save_dir: str | Path | None,
    snapshot_dir: str | Path,
    names: list[str] | tuple[str, ...] | None,
    dry_run: bool = False,
) -> list[str]:
    if not names:
        return []
    src = Path(snapshot_dir)
    for name in names:
        if Path(name).name != name:
            raise SystemExit(f"license_files entry {name!r} is not a bare filename — refusing")
        if not (src / name).is_file():
            raise SystemExit(f"license file {name!r} missing from base snapshot {src}")
        if (src / name).stat().st_size == 0:
            raise SystemExit(f"license file {name!r} in base snapshot {src} is EMPTY")
    if dry_run or save_dir is None:
        return list(names)
    d = Path(save_dir)
    stamped: list[str] = []
    for name in names:
        data = (src / name).read_bytes()
        fd, tmp = tempfile.mkstemp(dir=d, prefix=name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, d / name)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        if (d / name).read_bytes() != data:
            raise RuntimeError(f"license file {name} failed byte verification in {d}")
        stamped.append(name)
    return stamped


def base_generation_config(hub_id: str | Path) -> dict[str, Any]:
    path = Path(hub_id) / "generation_config.json"
    if not path.exists():
        if Path(hub_id).is_dir():
            return {}
        from huggingface_hub import hf_hub_download

        try:
            path = Path(hf_hub_download(str(hub_id), "generation_config.json"))
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                return {}
            raise RuntimeError(f"could not read {hub_id}/generation_config.json: {e}") from e
    try:
        gc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"unreadable {path}: {e}") from e
    if not isinstance(gc, dict):
        raise RuntimeError(f"{path} is not a JSON object ({type(gc).__name__})")
    return {k: v for k, v in gc.items() if not k.startswith("_") and k != "transformers_version"}


def build_generation_config(
    processor: Any,
    info: dict[str, Any] | None,
    hub_id: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tok = processor.tokenizer
    base = base_generation_config(hub_id)
    params = serving_params(info)
    params.update(overrides or {})
    gc = {k: v for k, v in base.items() if k not in _SAMPLING_KEYS}
    gc.update(params)
    gc["do_sample"] = True
    unk = getattr(tok, "unk_token_id", None)
    eos_list: list[int] = []
    for t in (
        tok.eos_token_id,
        tok.convert_tokens_to_ids("<|im_end|>"),
        tok.convert_tokens_to_ids("<|endoftext|>"),
    ):
        if t is not None and t != unk and t not in eos_list:
            eos_list.append(t)
    base_eos = gc.get("eos_token_id")
    merged = list(base_eos) if isinstance(base_eos, list) else ([] if base_eos is None else [base_eos])
    special_ids = set(tok.all_special_ids)
    merged = [t for t in merged if t in special_ids and tok.convert_ids_to_tokens(t) is not None]
    for t in eos_list:
        if t not in merged:
            merged.append(t)
    if not merged:
        raise SystemExit(
            "no stop tokens resolvable: tokenizer has no eos/<|im_end|>/<|endoftext|> and base card has only stale ids"
        )
    gc["eos_token_id"] = merged if len(merged) > 1 else merged[0]
    pad = tok.pad_token_id
    if pad is None:
        pad = base.get("pad_token_id", tok.eos_token_id)
    gc["pad_token_id"] = pad
    return gc


def stamp_vllm_autoload(save_dir: str | Path) -> None:
    """Stamp `vllm_warp_quant.py`, `auto_map` hooks, and canonical config normalizations."""
    save_dir = Path(save_dir)
    plugin_src = Path(__file__).parent / "vllm_plugin.py"
    if plugin_src.is_file():
        shutil.copyfile(plugin_src, save_dir / "vllm_warp_quant.py")

    cfg_path = save_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        is_dflash = cfg.get("architectures") == ["DFlash2DraftModel"]

        def _norm_re(val: Any) -> Any:
            if isinstance(val, str):
                if val.startswith("re:^layers\\."):
                    prefix = "re:.*" if is_dflash else "re:.*(?<!mtp\\.)"
                    return prefix + val[4:]
                if val.startswith("re:^mtp\\."):
                    return "re:.*" + val[4:]
                return val
            if isinstance(val, list):
                return [_norm_re(x) for x in val]
            if isinstance(val, dict):
                return {k: _norm_re(v) for k, v in val.items()}
            return val

        qcfg = cfg.get("quantization_config")
        if isinstance(qcfg, dict):
            cfg["quantization_config"] = _norm_re(qcfg)
            ple_qcfg = cfg["quantization_config"].get("ple_quantization")
            if isinstance(ple_qcfg, dict) and isinstance(cfg.get("text_config"), dict):
                cfg["text_config"]["ple_quantization"] = dict(ple_qcfg)

        model_type = cfg.get("model_type")
        if model_type != "qwen4_exp":
            am = dict(cfg.get("auto_map") or {})
            am["AutoConfig"] = "vllm_warp_quant.WarpQuantConfig"
            cfg["auto_map"] = am

        tmp = cfg_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(cfg_path)

    pc_path = save_dir / "processor_config.json"
    if pc_path.is_file():
        pc = json.loads(pc_path.read_text(encoding="utf-8"))
        am = dict(pc.get("auto_map") or {})
        am["AutoProcessor"] = "vllm_warp_quant.WarpQuantProcessor"
        pc["auto_map"] = am
        tmp = pc_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(pc_path)

        # Ensure standalone preprocessor_config.json & video_preprocessor_config.json exist
        # when transformers 5.x nests them inside processor_config.json
        img_proc = pc.get("image_processor")
        pre_path = save_dir / "preprocessor_config.json"
        if isinstance(img_proc, dict) and not pre_path.is_file():
            pre_obj = dict(img_proc)
            if "processor_class" in pc:
                pre_obj.setdefault("processor_class", pc["processor_class"])
            pre_path.write_text(json.dumps(pre_obj, indent=2) + "\n", encoding="utf-8")

        vid_proc = pc.get("video_processor")
        vpre_path = save_dir / "video_preprocessor_config.json"
        if isinstance(vid_proc, dict) and not vpre_path.is_file():
            vpre_obj = dict(vid_proc)
            if "processor_class" in pc:
                vpre_obj.setdefault("processor_class", pc["processor_class"])
            vpre_path.write_text(json.dumps(vpre_obj, indent=2) + "\n", encoding="utf-8")

    tc_path = save_dir / "tokenizer_config.json"
    if tc_path.is_file():
        tc = json.loads(tc_path.read_text(encoding="utf-8"))
        am = dict(tc.get("auto_map") or {})
        am["AutoTokenizer"] = [
            "vllm_warp_quant.WarpQuantTokenizer",
            "vllm_warp_quant.WarpQuantTokenizer",
        ]
        tc["auto_map"] = am
        tmp = tc_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(tc_path)


def stamp_release_template(save_dir: str | Path) -> str:
    save_dir = Path(save_dir)
    if not RELEASE_TEMPLATE.is_file():
        raise FileNotFoundError(f"release template missing: {RELEASE_TEMPLATE}")
    got = hashlib.sha256(RELEASE_TEMPLATE.read_bytes()).hexdigest()
    if got != RELEASE_TEMPLATE_SHA:
        raise ValueError(f"release_chat_template.jinja sha256 {got} != pinned {RELEASE_TEMPLATE_SHA}")
    shutil.copyfile(RELEASE_TEMPLATE, save_dir / "chat_template.jinja")
    tc_path = save_dir / "tokenizer_config.json"
    if tc_path.is_file():
        tc = json.loads(tc_path.read_text(encoding="utf-8"))
        if "chat_template" in tc:
            tc.pop("chat_template")
            tmp = tc_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(tc, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(tc_path)
    stamp_vllm_autoload(save_dir)
    shipped = save_dir / "chat_template.jinja"
    if hashlib.sha256(shipped.read_bytes()).hexdigest() != RELEASE_TEMPLATE_SHA:
        raise RuntimeError(f"shipped template sha mismatch: {shipped}")
    if tc_path.is_file() and "chat_template" in json.loads(tc_path.read_text(encoding="utf-8")):
        raise RuntimeError(f"tokenizer_config.json still embeds a template: {save_dir}")
    return RELEASE_TEMPLATE_SHA


def check_template_renders(save_dir: str | Path) -> str:
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(save_dir))
    text = proc.apply_chat_template(
        [{"role": "user", "content": "ping"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if "<|im_start|>" not in text:
        raise RuntimeError(f"shipped template did not render a chat prompt: {text!r}")
    tok = proc.tokenizer
    ids = tok(text, add_special_tokens=False)["input_ids"]
    unk = tok.unk_token_id
    if unk is not None and unk in ids:
        raise RuntimeError(f"shipped template produced unknown tokens: {save_dir}")
    return text
