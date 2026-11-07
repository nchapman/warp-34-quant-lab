"""Calibration datasets, text/MTP/Vision probe loaders, and v2 vision fixture materializer."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from warp_quant.lablog import log_event

MAX_LEN = 4096
CHUNK = 512

CASCADE_WEIGHTS: dict[str, float] = {
    "chat": 0.586,
    "math": 0.185,
    "science": 0.114,
    "instruction_following": 0.052,
    "conversational_agent": 0.035,
    "terminal_agent": 0.020,
    "swe": 0.006,
}
AGG_WEIGHTS: dict[str, float] = {
    **{
        domain: 0.9 * weight / math.fsum(CASCADE_WEIGHTS.values())
        for domain, weight in CASCADE_WEIGHTS.items()
    },
    "ML": 0.10,
}
WEIGHTED_KL_PROTOCOL = "cascade90-ml10-language-macro"

SOURCES_V2_CORE = (
    (
        "chartqa",
        "HuggingFaceM4/ChartQA",
        "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
        "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet",
    ),
    (
        "textvqa",
        "lmms-lab/TextVQA",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "data/validation-00000-of-00003.parquet",
    ),
    (
        "mmstar",
        "Lin-Chen/MMStar",
        "bc98d668301da7b14f648724866e57302778ab27",
        "mmstar.parquet",
    ),
    (
        "ocrbench",
        "echo840/OCRBench",
        "92a54bd1384387c178d5a07140a2d85e0a3d12e1",
        "data/test-00000-of-00001.parquet",
    ),
)

SOURCES_V2_GUARD = (
    (
        "screenspot",
        "rootsautomation/ScreenSpot",
        "0be08781e2e188582f6131625ae1598d443b4d5d",
        "data/test-00000-of-00003.parquet",
    ),
)

OCR_V2_BUCKET_MAP = {
    "docVQA": "docvqa",
    "infographicVQA": "infographic_vqa",
    "FUNSD": "funsd_forms",
    "SROIE": "kie_receipts_labels",
    "POIE": "kie_receipts_labels",
    "ocrVQA": "ocrvqa_scene",
    "ESTVQA": "ocrvqa_scene",
    "STVQA": "ocrvqa_scene",
}

MMSTAR_V2_BUCKET_MAP = {
    "coarse perception": "mmstar_perception",
    "fine-grained perception": "mmstar_perception",
    "instance reasoning": "mmstar_reasoning",
    "logical reasoning": "mmstar_reasoning",
    "math": "mmstar_math",
    "science & technology": "mmstar_science",
}

V2_IN_DIST_DOMAINS = (
    "chartqa_human",
    "chartqa_machine",
    "docvqa",
    "funsd_forms",
    "infographic_vqa",
    "kie_receipts_labels",
    "mmstar_math",
    "mmstar_perception",
    "mmstar_reasoning",
    "mmstar_science",
    "ocrvqa_scene",
    "textvqa",
)

V2_GUARD_DOMAINS = (
    "ocrbench_artistic_math",
    "screenspot_ios",
    "screenspot_macos_icon",
    "screenspot_macos_text",
    "screenspot_windows_icon",
    "screenspot_windows_text",
)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_messages(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for message in conversation:
        msg: dict[str, Any] = {"role": message["role"], "content": message.get("content") or ""}
        if message.get("tool_calls"):
            tool_calls = []
            for call in message["tool_calls"]:
                call = dict(call.get("function") or {})
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call["arguments"] = args
                tool_calls.append(call)
            msg["tool_calls"] = tool_calls
        messages.append(msg)
    return messages


def load_jsonl_calib(path: str | Path, num_samples: int) -> list[list[dict[str, Any]]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages") or r.get("conversations")
            rows.append(normalize_messages(msgs))
    return rows[:num_samples]


def get_calibration_dataset(
    processor: Any,
    calib: str | Path,
    num_samples: int,
    max_seq_length: int,
    window: str = "head",
    seed: int = 0,
) -> Dataset:
    tokenizer = processor.tokenizer
    path = Path(calib)
    conversations = load_jsonl_calib(path, num_samples)

    def tokenize(sample: dict[str, Any], index: int) -> dict[str, Any]:
        try:
            text = processor.apply_chat_template(normalize_messages(sample["messages"]), tokenize=False)
        except Exception:
            return {"input_ids": [], "attention_mask": []}
        if window == "random":
            rng = random.Random(hashlib.sha256(text.encode()).digest()[:8] + seed.to_bytes(4, "big"))
            enc = tokenizer(text, padding=False, truncation=False, add_special_tokens=False)
            ids = enc["input_ids"]
            if len(ids) > max_seq_length:
                off = rng.randrange(0, len(ids) - max_seq_length + 1)
                ids = ids[off : off + max_seq_length]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
            return {"input_ids": ids, "attention_mask": enc["attention_mask"]}
        tokenizer.truncation_side = "left" if window == "tail" else "right"
        return tokenizer(
            text,
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = Dataset.from_list([{"messages": m} for m in conversations])
    ds = ds.map(tokenize, with_indices=True, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    return ds


def aggregate_weighted_kl(domains: dict[str, dict[str, Any]]) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for domain, stats in domains.items():
        group = "ML" if domain.startswith("belebele") else domain
        if group not in AGG_WEIGHTS:
            continue
        if stats["kl_n"] <= 0:
            raise ValueError(f"domain {domain!r} has no KL tokens")
        groups[group].append(stats["kl"] / stats["kl_n"])
    if not groups:
        raise ValueError("weighted KL requires at least one recognized domain")
    total_weight = math.fsum(AGG_WEIGHTS[group] for group in groups)
    return (
        math.fsum(AGG_WEIGHTS[group] * math.fsum(means) / len(means) for group, means in groups.items())
        / total_weight
    )


def load_batches(
    processor: Any,
    path: str | Path,
    limit: int | None = None,
) -> list[tuple[str, torch.Tensor]]:
    batches: list[tuple[str, torch.Tensor]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages") or row.get("conversations")
            domain = row.get("domain", "all")
            text = processor.apply_chat_template(msgs, tokenize=False)
            enc = processor.tokenizer(
                text,
                max_length=MAX_LEN,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            batches.append((domain, enc["input_ids"]))
            if limit and len(batches) >= limit:
                break
    return batches


class FixedInputMTP(nn.Module):
    """Thin fusion wrapper around Qwen decoder/norm/RoPE modules for Native MTP evaluation."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        config = copy.deepcopy(config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config._attn_implementation = "eager"
        h, eps = config.hidden_size, config.rms_norm_eps
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, layer_idx=0)])
        self.norm = Qwen3_5RMSNorm(h, eps=eps)
        self.rotary = Qwen3_5TextRotaryEmbedding(config)

    def forward(
        self,
        hidden: torch.Tensor,
        embedded: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape != embedded.shape or positions.shape != (3, hidden.shape[0], hidden.shape[1]):
            raise ValueError("MTP hidden/embedding/position alignment mismatch")
        x = self.fc(
            torch.cat([self.pre_fc_norm_embedding(embedded), self.pre_fc_norm_hidden(hidden)], dim=-1)
        )
        length = x.shape[1]
        mask = torch.zeros((1, 1, length, length), device=x.device, dtype=x.dtype)
        mask.masked_fill_(
            torch.ones((length, length), device=x.device, dtype=torch.bool).triu(1), float("-inf")
        )
        x = self.layers[0](
            x,
            position_embeddings=self.rotary(x, positions),
            attention_mask=mask,
            past_key_values=None,
            use_cache=False,
        )
        return self.norm(x)


def align_mtp(
    ids: torch.Tensor,
    postnorm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 3 or postnorm.shape[:2] != ids.shape:
        raise ValueError("requires one unpadded sequence >=3 tokens with matching postnorm states")
    positions = torch.arange(ids.shape[1] - 2, device=ids.device).view(1, 1, -1).expand(3, 1, -1)
    return postnorm[:, :-2].detach().clone(), ids[:, 1:-1].clone(), positions, ids[:, 2:].clone()


def compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.shape[:-1] != labels.shape:
        raise ValueError("logits/label shape mismatch")
    p, q = F.log_softmax(reference.float(), -1), F.log_softmax(candidate.float(), -1)
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("nonfinite logits")
    kl = (p.exp() * (p - q)).sum(-1).flatten()
    tv = (0.5 * (p.exp() - q.exp()).abs().sum(-1)).flatten()
    return {
        "kl": kl.cpu().tolist(),
        "tv": tv.cpu().tolist(),
        "same_top1": int((reference.argmax(-1) == candidate.argmax(-1)).sum()),
        "reference_nll_sum": float(-p.gather(-1, labels.unsqueeze(-1)).sum()),
        "candidate_nll_sum": float(-q.gather(-1, labels.unsqueeze(-1)).sum()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty probe")
    kl = torch.tensor([x for r in rows for x in r["kl"]], dtype=torch.float64)
    tv = torch.tensor([x for r in rows for x in r["tv"]], dtype=torch.float64)
    if not len(kl):
        raise ValueError("empty scored tokens")
    return {
        "rows": len(rows),
        "tokens": len(kl),
        "mean_kl": kl.mean().item(),
        "median_kl": kl.median().item(),
        "p99_kl": torch.quantile(kl, 0.99).item(),
        "max_kl": kl.max().item(),
        "mean_tv": tv.mean().item(),
        "top1_agreement": sum(r["same_top1"] for r in rows) / len(kl),
        "reference_nll": sum(r["reference_nll_sum"] for r in rows) / len(kl),
        "candidate_nll": sum(r["candidate_nll_sum"] for r in rows) / len(kl),
    }


def feature_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("invalid encoder outputs")
    a_f, b_f = a.float(), b.float()
    error = (a_f - b_f).square().sum().item()
    energy = a_f.square().sum().item()
    if energy == 0:
        raise ValueError("zero-energy encoder reference; relative error is undefined")
    return {
        "elements": a.numel(),
        "squared_error": error,
        "reference_energy": energy,
        "relative_rmse": (error / energy) ** 0.5,
    }


def encode_image_row(
    processor: Any,
    data_dir: str | Path,
    row: dict[str, Any],
    max_pixels: int,
    blank: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    image_path = Path(data_dir) / row["image"]
    if sha256(image_path) != row["image_sha256"]:
        raise ValueError("image checksum mismatch")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    if blank:
        image = Image.new("RGB", image.size, (127, 127, 127))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": row["question"] + "\nAnswer with only the answer, without explanation.",
                },
            ],
        }
    ]
    prefix = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    kwargs = dict(
        images=[image],
        return_tensors="pt",
        add_special_tokens=False,
        min_pixels=128 * 128,
        max_pixels=max_pixels,
    )
    prompt = processor(text=[prefix], **kwargs)
    full = processor(text=[prefix + row["answer"]], **kwargs)
    count = prompt["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :count], prompt["input_ids"]):
        raise ValueError("answer changed prefix tokenization")
    labels = full["input_ids"][:, count:].clone()
    if not labels.numel():
        raise ValueError("empty answer tokens")
    keep = torch.arange(count - 1, full["input_ids"].shape[1] - 1)
    return full, keep, labels


def device_inputs(inputs: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
        for k, v in inputs.items()
    }


def select_text_rows(path: str | Path, per_domain: int) -> list[tuple[int, str, list[dict[str, Any]]]]:
    counts: Counter[str] = Counter()
    result: list[tuple[int, str, list[dict[str, Any]]]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        domain = row.get("domain", "all")
        if counts[domain] < per_domain:
            result.append((index, domain, row.get("messages") or row["conversations"]))
            counts[domain] += 1
    if not result or min(counts.values()) != per_domain:
        raise ValueError("insufficient text fixtures per domain")
    return result


def _select_v2_split(
    candidates: list[tuple[str, str, int, bytes, str, list[str]]],
    seed_tag: str,
    budgets: dict[str, int],
    seen: set[str],
    min_dim: int,
) -> list[tuple[str, str, int, str, bytes, int, int, str, list[str]]]:
    ranked = []
    for dom, cat, idx, blob, q, ans in candidates:
        digest = hashlib.sha256(blob).hexdigest()
        rk = hashlib.sha256(f"{seed_tag}:{dom}:{digest}:{idx}".encode()).hexdigest()
        ranked.append((rk, dom, cat, idx, digest, blob, q, ans))
    counts: Counter[str] = Counter()
    picked = []
    for _, dom, cat, idx, digest, blob, q, ans in sorted(ranked):
        if digest in seen or counts[dom] >= budgets.get(dom, 0):
            continue
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
            image.verify()
        if min(width, height) < min_dim:
            continue
        counts[dom] += 1
        seen.add(digest)
        picked.append((dom, cat, idx, digest, blob, width, height, q, ans))
    if dict(counts) != budgets:
        raise ValueError(f"incomplete v2 {seed_tag} selection: {dict(counts)} != {budgets}")
    return picked


def prepare_v2_vision_split(output: str | Path, split: str = "probe") -> Path:
    if split not in ("probe", "calibration", "guard"):
        raise ValueError(f"unsupported v2 split: {split!r}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()

    sources = SOURCES_V2_CORE + (SOURCES_V2_GUARD if split == "guard" else ())
    provenance = []
    source_paths: dict[str, str] = {}
    for name, repo, revision, file in sources:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename=file)
        provenance.append(
            {"domain": name, "repo": repo, "revision": revision, "file": file, "sha256": sha256(path)}
        )
        source_paths[name] = path

    in_dist: list[tuple[str, str, int, bytes, str, list[str]]] = []
    guard_cands: list[tuple[str, str, int, bytes, str, list[str]]] = []

    for idx, row in enumerate(pq.ParquetFile(source_paths["chartqa"]).read().to_pylist()):
        blob = row["image"]["bytes"]
        ans = [str(a).strip() for a in (row["label"] or []) if isinstance(a, str) and str(a).strip()]
        q = (row["query"] or "").strip()
        if ans and q:
            dom = "chartqa_human" if str(row["human_or_machine"]) == "0" else "chartqa_machine"
            in_dist.append((dom, f"chartqa:{row['human_or_machine']}", idx, blob, q, ans))

    for idx, row in enumerate(pq.ParquetFile(source_paths["textvqa"]).read().to_pylist()):
        blob = row["image"]["bytes"]
        ans = [str(a).strip() for a in (row["answers"] or []) if isinstance(a, str) and str(a).strip()]
        q = (row["question"] or "").strip()
        if ans and q:
            in_dist.append(("textvqa", "textvqa:all", idx, blob, q, ans))

    for idx, row in enumerate(pq.ParquetFile(source_paths["ocrbench"]).read().to_pylist()):
        ds = row["dataset"]
        blob = row["image"]["bytes"]
        ans = [str(a).strip() for a in (row["answer"] or []) if isinstance(a, str) and str(a).strip()]
        q = (row["question"] or "").strip()
        if not (ans and q):
            continue
        if ds in OCR_V2_BUCKET_MAP:
            in_dist.append((OCR_V2_BUCKET_MAP[ds], f"ocrbench:{ds}", idx, blob, q, ans))
        elif ds in ("WordArt", "HME100k"):
            guard_cands.append(("ocrbench_artistic_math", f"ocrbench:{ds}", idx, blob, q, ans))

    for idx, row in enumerate(pq.ParquetFile(source_paths["mmstar"]).read().to_pylist()):
        cat = row["category"]
        if cat not in MMSTAR_V2_BUCKET_MAP:
            continue
        img = row["image"]
        blob = img if isinstance(img, bytes) else img["bytes"]
        ans = [str(row["answer"]).strip()] if row.get("answer") and str(row["answer"]).strip() else []
        q = (row["question"] or "").strip()
        if ans and q:
            in_dist.append((MMSTAR_V2_BUCKET_MAP[cat], f"mmstar:{cat}", idx, blob, q, ans))

    seen: set[str] = set()
    probe_budgets = {d: 8 for d in V2_IN_DIST_DOMAINS}
    calib_budgets = {d: 10 for d in V2_IN_DIST_DOMAINS}
    probe_picked = _select_v2_split(in_dist, "vision-probe-v2-0", probe_budgets, seen, min_dim=128)

    if split == "probe":
        selected = probe_picked
        selection_desc = (
            "suite v2 probe: hash seed vision-probe-v2-0; 12 domains x 8 images (96 total, min_dim>=128)"
        )
    else:
        calib_picked = _select_v2_split(in_dist, "vision-calib-v2-1", calib_budgets, seen, min_dim=128)
        if split == "calibration":
            selected = calib_picked
            selection_desc = (
                "suite v2 calibration: hash seed vision-calib-v2-1 (disjoint from vision-probe-v2-0); "
                "12 domains x 10 images (120 total, min_dim>=128)"
            )
        else:
            for idx, row in enumerate(pq.ParquetFile(source_paths["screenspot"]).read().to_pylist()):
                img = row["image"]
                blob = img if isinstance(img, bytes) else img["bytes"]
                src = str(row["data_source"] or "").strip()
                dtype = str(row["data_type"] or "").strip()
                inst = str(row["instruction"] or "").strip()
                if not (src and dtype and inst):
                    continue
                dom = f"screenspot_{src}_{dtype}" if src in ("macos", "windows") else "screenspot_ios"
                q = (
                    f"In this {src} interface screenshot, is the UI element for '{inst}' "
                    f"a text element or an icon?\nOptions: text, icon"
                )
                guard_cands.append((dom, f"screenspot:{src}:{dtype}", idx, blob, q, [dtype]))
            guard_budgets = {d: 8 for d in V2_GUARD_DOMAINS}
            selected = _select_v2_split(guard_cands, "vision-guard-v2-0", guard_budgets, seen, min_dim=96)
            selection_desc = (
                "suite v2 guard: hash seed vision-guard-v2-0 (disjoint from probe and calibration); "
                "6 OOD domains x 8 images (48 total, min_dim>=96)"
            )

    records = []
    for dom, cat, idx, digest, blob, width, height, q, ans in selected:
        filename = f"images/{digest}.image"
        (out / filename).write_bytes(blob)
        records.append(
            {
                "domain": dom,
                "category": cat,
                "source_row": idx,
                "image": filename,
                "image_sha256": digest,
                "width": width,
                "height": height,
                "question": q,
                "answers": ans,
                "answer": Counter(ans).most_common(1)[0][0],
            }
        )
    log_event("SOURCE_SELECTED", suite="v2", split=split, rows=len(records))
    rows_path = out / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    manifest = {
        "suite": "v2",
        "split": split,
        "sources": provenance,
        "rows": len(records),
        "rows_sha256": sha256(rows_path),
        "selection": selection_desc,
        "calibration": split == "calibration",
        "guard": split == "guard",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
