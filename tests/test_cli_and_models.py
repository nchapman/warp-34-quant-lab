"""Unit tests for warp_quant CLI, model registry, and release stamping."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from warp_quant.cli import build_parser, main
from warp_quant.models import MODELS, quant_recipe, resolve, serving_params
from warp_quant.stamp import RELEASE_TEMPLATE_SHA, stamp_release_template


def test_models_registry_defaults_to_w4a16_hgptq():
    for key in ("0.8b", "2b", "4b", "9b", "27b", "35b", "flash-next"):
        hub_id, info = resolve(key)
        assert hub_id == MODELS[key]["hub"]
        assert info["quant_repo"].endswith("-W4A16-HGPTQ")
        rec = quant_recipe(info)
        assert rec["algo"] == "hgptq"
        assert rec["group_size"] == 64
        assert rec["hadamard_block_size"] == 64
        assert rec["actorder"] == "static"
        assert rec["quantize_mtp"] is True
        assert rec["quantize_vision"] is True

    _, mimo_info = resolve("9b-mimo")
    assert serving_params(mimo_info)["temperature"] == 0.6
    _, base_info = resolve("35b")
    assert serving_params(base_info) == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def test_cli_parser_defaults(capsys):
    parser = build_parser()
    q_args = parser.parse_args(["quantize", "--model", "flash-next", "--output-dir", "/tmp/out"])
    assert q_args.model == "flash-next"
    assert q_args.group_size == 64
    assert q_args.hadamard_block_size == 64
    assert q_args.ple_group_size == 32
    assert q_args.ple_cd_passes == 1
    assert q_args.actorder == "static"
    assert q_args.quantize_mtp is True
    assert q_args.quantize_vision is True
    assert q_args.component_quant_mode == "gptq"

    p_args = parser.parse_args(["probe", "--model", "35b", "--quant", "/tmp/out", "--eval-mtp"])
    assert p_args.model == "35b"
    assert p_args.eval_mtp is True
    assert p_args.eval_sets == ["data/probe-v4.jsonl", "data/probe-guard.jsonl"]

    main(["list-models"])
    out = capsys.readouterr().out
    assert "Qwen/Qwen3.6-35B-A3B" in out
    assert "Qwen/Qwen3.8-Flash-Next" in out


def test_stamp_release_template_strips_embedded_chat_template():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "tokenizer_config.json").write_text(
            json.dumps({"eos_token": "<|im_end|>", "chat_template": "OLD_EMBEDDED"}),
            encoding="utf-8",
        )
        sha = stamp_release_template(d)
        assert sha == RELEASE_TEMPLATE_SHA
        assert (d / "chat_template.jinja").is_file()
        tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert "chat_template" not in tc
