"""Qwen 3.5 / 3.6 / 3.8 model registry and W4A16-HGPTQ recipe defaults."""

from __future__ import annotations

from typing import Any

# Canonical thinking-mode sampling trio across Qwen3.5 / 3.6 / 3.8 cards
SERVING_PARAMS: dict[str, Any] = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}

# Champion W4A16-HGPTQ recipe defaults:
# - group_size=64, actorder="static", hadamard_block_size=64 (online H_64 input rotation)
# - quantize_mtp=True, quantize_vision=True, component_quant_mode="gptq"
# - ple_group_size=32, ple_cd_passes=1 (4-bit Custom-LUT4 W_kv-HGPTQ for Layer-1 PLE)
QUANT_RECIPE: dict[str, Any] = {
    "algo": "hgptq",
    "calib": "data/calib-cascade-v1.jsonl",
    "num_samples": 999,
    "max_seq_length": 4096,
    "window": "head",
    "group_size": 64,
    "hadamard_block_size": 64,
    "actorder": "static",
    "dampening_frac": 0.01,
    "ple_group_size": 32,
    "ple_cd_passes": 1,
    "quantize_mtp": True,
    "quantize_vision": True,
    "component_quant_mode": "gptq",
}

MODELS: dict[str, dict[str, Any]] = {
    "0.8b": {
        "hub": "Qwen/Qwen3.5-0.8B",
        "generation": "3.5",
        "arch": "qwen3_5",
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 3584,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Qwen3.5-0.8B-W4A16-HGPTQ",
    },
    "2b": {
        "hub": "Qwen/Qwen3.5-2B",
        "generation": "3.5",
        "arch": "qwen3_5",
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 6144,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Qwen3.5-2B-W4A16-HGPTQ",
    },
    "4b": {
        "hub": "Qwen/Qwen3.5-4B",
        "generation": "3.5",
        "arch": "qwen3_5",
        "num_hidden_layers": 32,
        "hidden_size": 2560,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 9216,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Qwen3.5-4B-W4A16-HGPTQ",
    },
    "9b": {
        "hub": "Qwen/Qwen3.5-9B",
        "generation": "3.5",
        "arch": "qwen3_5",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Qwen3.5-9B-W4A16-HGPTQ",
    },
    "27b": {
        "hub": "Qwen/Qwen3.8-27B",
        "generation": "3.8",
        "arch": "qwen3_5",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Qwen3.8-27B-W4A16-HGPTQ",
    },
    "35b": {
        "hub": "Qwen/Qwen3.6-35B-A3B",
        "generation": "3.6",
        "arch": "qwen3_5_moe",
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ",
    },
    "flash-next": {
        "hub": "Qwen/Qwen3.8-Flash-Next",
        "generation": "3.8",
        "arch": "qwen4_exp",
        "num_hidden_layers": 48,
        "hidden_size": 2560,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "ple_layer_ids": [2],
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_recipe": {
            "ple_group_size": 32,
            "ple_cd_passes": 1,
        },
        "quant_repo": "nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ",
    },
    "9b-ornith": {
        "hub": "ornith-ai/Ornith-1.5-9B",
        "generation": "ornith-1.5",
        "arch": "qwen3_5",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Ornith-1.5-9B-W4A16-HGPTQ",
    },
    "35b-ornith": {
        "hub": "ornith-ai/Ornith-1.5-35B-A3B",
        "generation": "ornith-1.5",
        "arch": "qwen3_5_moe",
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "quant_repo": "nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ",
    },
    "9b-mimo": {
        "hub": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
        "hub_revision": "2367e865d009c13ac81713a2878291d33ab28177",
        "generation": "xiaomi-mimo",
        "arch": "qwen3_5",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "sampling": {"temperature": 0.6},
        "quant_repo": "nchapman/MiMo-V2.6-Distill-Qwen-9B-W4A16-HGPTQ",
    },
    "27b-swift": {
        "hub": "ukisai/Swift-1.5-Qwen3.8-27b",
        "hub_revision": "b4c84d42903a8646b25857eb2827288d92ed85a4",
        "generation": "swift-1.5",
        "arch": "qwen3_5",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "license_name": "swift-open-license-1.0",
        "license_link": (
            "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/"
            "b4c84d42903a8646b25857eb2827288d92ed85a4/LICENSE"
        ),
        "license_files": ["LICENSE", "NOTICE", "LICENSE-APACHE-2.0"],
        "quant_repo": "nchapman/Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ",
    },
    "0.8b-abliterated": {
        "hub": "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated",
        "hub_revision": "4813135658fe51b2e535b8e77927d2f23909ce42",
        "generation": "huihui-ai",
        "arch": "qwen3_5",
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 3584,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Huihui-Qwen3.5-0.8B-abliterated-W4A16-HGPTQ",
    },
    "2b-abliterated": {
        "hub": "huihui-ai/Huihui-Qwen3.5-2B-abliterated",
        "hub_revision": "b2e291a65f29a9b148981fa5299caea5d35bd4c8",
        "generation": "huihui-ai",
        "arch": "qwen3_5",
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 6144,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Huihui-Qwen3.5-2B-abliterated-W4A16-HGPTQ",
    },
    "4b-abliterated": {
        "hub": "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        "hub_revision": "5581467dfd52bf338c782006a6cdce05c42594be",
        "generation": "huihui-ai",
        "arch": "qwen3_5",
        "num_hidden_layers": 32,
        "hidden_size": 2560,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 9216,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "processor_hub_fallback": "Qwen/Qwen3.5-4B",
        "processor_hub_fallback_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "quant_repo": "nchapman/Huihui-Qwen3.5-4B-abliterated-W4A16-HGPTQ",
    },
    "9b-abliterated": {
        "hub": "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
        "hub_revision": "05b9e7c9b978ba29bdb8f50a49c30e4b91183339",
        "generation": "huihui-ai",
        "arch": "qwen3_5",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Huihui-Qwen3.5-9B-abliterated-W4A16-HGPTQ",
    },
    "27b-abliterated": {
        "hub": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
        "hub_revision": "739e3c5b89849f6c238ce1e5b70008612ae42cdd",
        "generation": "huihui-ai",
        "arch": "qwen3_5",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Huihui-Qwen3.8-27B-abliterated-W4A16-HGPTQ",
    },
    "35b-abliterated": {
        "hub": "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated",
        "hub_revision": "8f0ee727aff5e771ea72466d64d13ecd851d2cc7",
        "generation": "huihui-ai",
        "arch": "qwen3_5_moe",
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "mask_token_id": 248077,
        "vision_tower": True,
        "mtp_num_hidden_layers": 1,
        "quant_repo": "nchapman/Huihui-Qwen3.6-35B-A3B-abliterated-W4A16-HGPTQ",
    },
}

FLASH_NEXT_INFO = MODELS["flash-next"]


def serving_params(info: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(SERVING_PARAMS)
    params.update((info or {}).get("sampling") or {})
    return params


def quant_recipe(info: dict[str, Any] | None) -> dict[str, Any]:
    recipe = dict(QUANT_RECIPE)
    recipe.update((info or {}).get("quant_recipe") or {})
    return recipe


def model_class(hub_id: str):
    from transformers import AutoConfig

    model_type = AutoConfig.from_pretrained(hub_id).model_type
    if model_type == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

        return Qwen3_5MoeForConditionalGeneration
    if model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration
    if model_type == "qwen4_exp":
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

        return Qwen4ExpForConditionalGeneration
    raise ValueError(f"unsupported model_type {model_type!r} for {hub_id}")


def get(key: str) -> dict[str, Any] | None:
    if key in MODELS:
        return MODELS[key]
    if key == "qwen3.8-flash-next":
        return MODELS["flash-next"]
    return None


def resolve(key: str) -> tuple[str, dict[str, Any]]:
    info = get(key)
    if info is not None:
        return info["hub"], info
    return key, {}
