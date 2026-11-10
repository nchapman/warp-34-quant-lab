"""Unit tests for Qwen/Qwen3.8-Flash-Next streaming W4A16-HGPTQ + 4-bit Custom-LUT4 PLE quantizer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from warp_quant.hadamard_gptq import _assert_packed_alignment
from warp_quant.models import model_class, quant_recipe, resolve
from warp_quant.streaming_moe import (
    PLE_CUSTOM_LUT4,
    Qwen4ExpMultiTokenPredictor,
    _build_lightweight_decoder_layer,
    _ensure_qwen4_exp_device_kernels,
    _forward_qsa_attention_fast,
    _load_decoder_layer_weights,
    _load_qwen4_exp_mtp_from_checkpoint,
    _quantize_decoder_layer_streaming,
    _stream_forward_qwen4_exp_states,
    _write_hot_expert_rankings,
    _write_qwen4_exp_quant_config,
    compute_ple_head_hessians,
    compute_ple_ngram_ids,
    dequantize_ple_rows_lut4,
    pack_ple_page_slotted_4k,
    quantize_ple_batch_gptq_lut4,
    unpack_ple_page_slotted_4k,
)


def _make_tiny_qwen4_exp_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    cfg = Qwen4ExpConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "layer_types": ["indexed_attention", "linear_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_compress_ratio": 1,
            "indexer_head_dim": 32,
            "indexer_budget": 64,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "hc_count": 4,
            "ple_layer_ids": [2],
            "ple_embed_dim": 128,
            "ple_conv_kernel": 2,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 64,
            "make_ngram_vocab_size_divisible_by": 4,
            "eos_token_id": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [4, 2, 2],
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 80,
            "num_heads": 4,
            "in_channels": 3,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 64,
        },
    )
    _ensure_qwen4_exp_device_kernels(cfg.text_config, torch.device("cpu"))
    return cfg


class TestQwen38FlashNextQuant(unittest.TestCase):
    def test_models_registry_and_model_class(self):
        hub_id, info = resolve("flash-next")
        self.assertEqual(hub_id, "Qwen/Qwen3.8-Flash-Next")
        self.assertEqual(info["arch"], "qwen4_exp")
        self.assertEqual(info["num_experts"], 512)
        self.assertEqual(info["ple_layer_ids"], [2])
        self.assertTrue(info["vision_tower"])
        recipe = quant_recipe(info)
        self.assertEqual(recipe["calib"], "data/calib-cascade-v1.jsonl")
        self.assertEqual(recipe["num_samples"], 999)
        self.assertEqual(recipe["max_seq_length"], 4096)
        self.assertEqual(recipe["group_size"], 64)
        self.assertEqual(recipe["hadamard_block_size"], 64)
        self.assertEqual(recipe["ple_group_size"], 32)
        self.assertEqual(recipe["ple_cd_passes"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            cls = model_class(str(p))
            self.assertEqual(cls.__name__, "Qwen4ExpForConditionalGeneration")

    def test_ple_custom_lut4_wkv_gptq_and_dequantize(self):
        torch.manual_seed(0)
        num_heads = 2
        head_dim = 64
        embed_dim = num_heads * head_dim
        out_k = 128
        out_v = 32

        # Synthesize correlated downstream key_proj / value_proj
        u_orth, _ = torch.linalg.qr(torch.randn(embed_dim, embed_dim))
        eig_profile = torch.linspace(0.2, 3.0, embed_dim)
        wk = torch.randn(out_k, embed_dim) @ (torch.diag(eig_profile) @ u_orth.T)
        wv = torch.randn(out_v, embed_dim) @ (torch.diag(eig_profile.flip(0)) @ u_orth.T)

        h_list, u_list = compute_ple_head_hessians(wk, wv, num_heads=num_heads, dampening_frac=0.01)
        self.assertEqual(len(h_list), num_heads)
        self.assertEqual(h_list[0].shape, (head_dim, head_dim))

        w_rows = torch.randn(256, head_dim) * 0.02
        lut = torch.tensor(PLE_CUSTOM_LUT4, dtype=torch.float32)

        # W_kv-HGPTQ + 1-pass CD
        q_gptq, packed_u8, scales_fp16 = quantize_ple_batch_gptq_lut4(
            w_rows, h_list[0], u_list[0], lut=lut, group_size=32, cd_passes=1
        )
        self.assertEqual(packed_u8.shape, (256, head_dim // 2))
        self.assertEqual(packed_u8.dtype, torch.uint8)
        self.assertEqual(scales_fp16.shape, (256, head_dim // 32))
        self.assertEqual(scales_fp16.dtype, torch.float16)

        deq_fp32 = dequantize_ple_rows_lut4(
            packed_u8, scales_fp16, lut, group_size=32, out_dtype=torch.float32
        )
        self.assertTrue(torch.allclose(q_gptq, deq_fp32, atol=1e-6))

        # Compare downstream weighted loss against RTN (identity Hessian, 0 CD passes)
        eye = torch.eye(head_dim)
        q_rtn, _, _ = quantize_ple_batch_gptq_lut4(w_rows, eye, eye, lut=lut, group_size=32, cd_passes=0)
        err_gptq = ((q_gptq - w_rows) @ h_list[0] * (q_gptq - w_rows)).sum().item()
        err_rtn = ((q_rtn - w_rows) @ h_list[0] * (q_rtn - w_rows)).sum().item()
        self.assertLess(err_gptq, err_rtn)

    def test_ple_page_slotted_4k_pack_unpack_roundtrip(self):
        torch.manual_seed(1)
        n_rows = 100  # 3 pages of 45 rows/page
        wp = torch.randint(0, 256, (n_rows, 80), dtype=torch.uint8)
        ws = torch.randn(n_rows, 5, dtype=torch.float16)
        pages = pack_ple_page_slotted_4k(wp, ws)
        self.assertEqual(pages.shape, (3, 4096))
        self.assertEqual(pages.dtype, torch.uint8)
        # Unused trailing 16 B per 4 KiB page must be zero
        self.assertEqual(int(pages[:, 4050:].abs().max().item()), 0)

        wp_out, ws_out = unpack_ple_page_slotted_4k(pages, num_rows=n_rows, head_dim=160, group_size=32)
        self.assertTrue(torch.equal(wp, wp_out))
        self.assertTrue(torch.equal(ws, ws_out))

    def test_compute_ple_ngram_ids_matches_transformers(self):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

        cfg = _make_tiny_qwen4_exp_config()
        emb = Qwen4ExpTextNGramEmbedding(
            cfg.text_config, cfg.text_config.ple_embed_dim, layer_idx=1, ple_layer_index=0
        ).eval()
        # Include eos_token_id=2 inside the token sequence to test segment boundary resets
        input_ids = torch.tensor([[5, 19, 2, 41, 88, 12, 2, 7], [3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)

        n_ids = compute_ple_ngram_ids(
            input_ids,
            emb.layer_multipliers,
            emb.ngram_heads_vocab_sizes,
            emb.ngram_heads_offsets,
            emb.ngram_size,
            emb.heads_per_ngram,
            emb.eos_token_id,
        )
        ref_out = emb(input_ids, past_key_values=None)
        gathered_out = emb.ngram_embedding(n_ids).flatten(-2)
        self.assertTrue(torch.equal(ref_out, gathered_out))

    def test_qsa_attention_fast_matches_indexed_attention(self):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextAttention,
            Qwen4ExpTextRotaryEmbedding,
        )

        torch.manual_seed(2)
        cfg = _make_tiny_qwen4_exp_config()
        attn = Qwen4ExpTextAttention(cfg.text_config, layer_idx=0).to(torch.bfloat16).eval()
        rotary = Qwen4ExpTextRotaryEmbedding(cfg.text_config)
        x = torch.randn(1, 12, cfg.text_config.hidden_size, dtype=torch.bfloat16)
        pos_ids = torch.arange(12).view(1, 1, -1).expand(3, 1, -1)
        pos_emb = rotary(x, pos_ids)
        causal = torch.tril(torch.ones((1, 1, 12, 12), dtype=torch.bool))

        ref_out, _ = attn(x, pos_emb, attention_mask=causal)
        fast_out = _forward_qsa_attention_fast(attn, x, pos_emb)
        self.assertEqual((ref_out - fast_out).abs().max().item(), 0.0)

        # Also test S > indexer.token_budget with compress_ratio=2, budget=4 -> vectorized causal indexer path
        cfg_long = _make_tiny_qwen4_exp_config()
        cfg_long.text_config.indexer_budget = 4
        cfg_long.text_config.indexer_compress_ratio = 2
        attn_long = Qwen4ExpTextAttention(cfg_long.text_config, layer_idx=0).to(torch.bfloat16).eval()
        x_long = torch.randn(1, 11, cfg_long.text_config.hidden_size, dtype=torch.bfloat16)
        pos_ids_long = torch.arange(11).view(1, 1, -1).expand(3, 1, -1)
        pos_emb_long = rotary(x_long, pos_ids_long)
        causal_long = torch.tril(torch.ones((1, 1, 11, 11), dtype=torch.bool))
        ref_long, _ = attn_long(x_long, pos_emb_long, attention_mask=causal_long)
        fast_long = _forward_qsa_attention_fast(attn_long, x_long, pos_emb_long)
        self.assertEqual((ref_long - fast_long).abs().max().item(), 0.0)

    def test_streaming_layer_and_mtp_quantize_and_forward_roundtrip(self):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpForConditionalGeneration,
            Qwen4ExpTextRotaryEmbedding,
        )

        torch.manual_seed(42)
        cfg = _make_tiny_qwen4_exp_config()
        model = Qwen4ExpForConditionalGeneration(cfg).to(torch.bfloat16).eval()
        mtp = Qwen4ExpMultiTokenPredictor(cfg.text_config, torch.device("cpu"))
        for p in mtp.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "base"
            quant_dir = Path(tmp) / "quant"
            base_dir.mkdir()
            quant_dir.mkdir()

            raw_cfg = cfg.to_dict()
            (base_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2) + "\n")

            # Build base safetensors shards matching Qwen/Qwen3.8-Flash-Next layout:
            # - 2D per-expert weights `mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`
            # - 2 PLE shards `layers.1.ple.ple_embedding.ngram_embedding.shard_{0,1}.weight`
            # - `mtp.*` weights
            base_tensors: dict[str, torch.Tensor] = {}
            for k, v in model.state_dict().items():
                if k.endswith("mlp.experts.gate_up_proj"):
                    pref = k[: -len("gate_up_proj")]
                    inter = v.shape[1] // 2
                    for e in range(v.shape[0]):
                        base_tensors[f"{pref}{e}.gate_proj.weight"] = v[e, :inter].contiguous()
                        base_tensors[f"{pref}{e}.up_proj.weight"] = v[e, inter:].contiguous()
                elif k.endswith("mlp.experts.down_proj"):
                    pref = k[: -len("down_proj")]
                    for e in range(v.shape[0]):
                        base_tensors[f"{pref}{e}.down_proj.weight"] = v[e].contiguous()
                elif k == "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight":
                    mid = v.shape[0] // 2
                    pref = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
                    base_tensors[f"{pref}.shard_0.weight"] = v[:mid].contiguous()
                    base_tensors[f"{pref}.shard_1.weight"] = v[mid:].contiguous()
                else:
                    base_tensors[k] = v.contiguous()

            for k, v in mtp.state_dict().items():
                if k == "layers.0.mlp.experts.gate_up_proj":
                    inter = v.shape[1] // 2
                    for e in range(v.shape[0]):
                        base_tensors[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = v[
                            e, :inter
                        ].contiguous()
                        base_tensors[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = v[
                            e, inter:
                        ].contiguous()
                elif k == "layers.0.mlp.experts.down_proj":
                    for e in range(v.shape[0]):
                        base_tensors[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = v[e].contiguous()
                else:
                    base_tensors[f"mtp.{k}"] = v.contiguous()

            save_file(base_tensors, str(base_dir / "model-00001-of-00001.safetensors"))
            wmap = {k: "model-00001-of-00001.safetensors" for k in base_tensors}
            (base_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 0}, "weight_map": wmap}, indent=2) + "\n"
            )

            # Verify streaming BF16 forward matches eager Qwen4ExpForConditionalGeneration forward
            calib_ids = [
                torch.randint(3, 250, (1, 16), dtype=torch.long),
                torch.randint(3, 250, (1, 16), dtype=torch.long),
            ]
            ref_logits_0 = model(input_ids=calib_ids[0]).logits
            stream_final, stream_pre_mixer, stream_head = _stream_forward_qwen4_exp_states(
                base_dir, cfg.text_config, calib_ids, torch.device("cpu"), return_pre_mixer=True
            )
            stream_logits_0 = stream_head(stream_final[0])
            self.assertTrue(torch.allclose(ref_logits_0, stream_logits_0, atol=2e-3, rtol=1e-2))

            # Run streaming W4A16-HGPTQ + Custom-LUT4 PLE quantization across layers 0, 1, and MTP
            device = torch.device("cpu")
            rotary = Qwen4ExpTextRotaryEmbedding(cfg.text_config)
            embed_w = model.model.language_model.embed_tokens.weight.detach()
            h_calib = []
            pos_list = []
            for ids in calib_ids:
                inp = F.embedding(ids, embed_w)
                pids = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
                h_calib.append(inp.repeat(1, 1, cfg.text_config.hc_count))
                pos_list.append(rotary(inp, pids))

            quant_wmap: dict[str, str] = {}
            embed_head = {
                k: v.clone()
                for k, v in base_tensors.items()
                if k.startswith(
                    (
                        "model.language_model.embed_tokens.",
                        "model.language_model.hyper_connection_mixer.",
                        "lm_head.",
                    )
                )
            }
            save_file(embed_head, str(quant_dir / "model-embed-head.safetensors"))
            for tk in embed_head:
                quant_wmap[tk] = "model-embed-head.safetensors"

            for l_idx in range(2):
                layer = _build_lightweight_decoder_layer(cfg.text_config, l_idx, device)
                l_pref = f"model.language_model.layers.{l_idx}"
                _load_decoder_layer_weights(layer, l_pref, base_dir, wmap, device)
                l_tensors, ple_wmap, _l_bytes, l_counts = _quantize_decoder_layer_streaming(
                    layer,
                    l_pref,
                    h_calib,
                    pos_list,
                    calib_ids,
                    base_dir=base_dir,
                    shard_map=wmap,
                    output_dir=quant_dir,
                    layer_idx=l_idx,
                    group_size=32,
                    hadamard_block_size=32,
                    actorder="static",
                    dampening_frac=0.01,
                    ple_group_size=16,
                    ple_cd_passes=1,
                )
                l_fname = f"model-layer-{l_idx:02d}.safetensors"
                save_file(l_tensors, str(quant_dir / l_fname))
                for tk in l_tensors:
                    quant_wmap[tk] = l_fname
                quant_wmap.update(ple_wmap)
                self.assertEqual(l_counts["routed_experts"], 12)  # 4 experts * 3 projs
                if l_idx == 1:
                    self.assertEqual(l_counts["ple_shards"], 2)

            # Quantize MTP
            mtp_loaded = _load_qwen4_exp_mtp_from_checkpoint(base_dir, cfg.text_config, device)
            h_mtp_calib = []
            mtp_pos_list = []
            mtp_ids_list = []
            for i in range(len(calib_ids)):
                next_ids = calib_ids[i][:, 1:]
                next_emb = F.embedding(next_ids, embed_w)
                h_in = mtp_loaded.prepare_layer_input(h_calib[i][:, :-1, :], next_emb)
                pids_m = torch.arange(next_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
                h_mtp_calib.append(h_in)
                mtp_pos_list.append(mtp_loaded.rotary_emb(next_emb, pids_m))
                mtp_ids_list.append(next_ids)

            mtp_tensors, _, _, mtp_counts = _quantize_decoder_layer_streaming(
                mtp_loaded.layers[0],
                "mtp.layers.0",
                h_mtp_calib,
                mtp_pos_list,
                mtp_ids_list,
                base_dir=base_dir,
                shard_map=wmap,
                output_dir=quant_dir,
                layer_idx=0,
                group_size=32,
                hadamard_block_size=32,
                actorder="static",
                dampening_frac=0.01,
            )
            self.assertEqual(mtp_counts["attn_linears"], 4)
            self.assertEqual(mtp_counts["routed_experts"], 12)
            for name, param in mtp_loaded.named_parameters():
                if not name.startswith("layers."):
                    mtp_tensors[f"mtp.{name}"] = param.detach().clone()
            save_file(mtp_tensors, str(quant_dir / "model-mtp.safetensors"))
            for tk in mtp_tensors:
                quant_wmap[tk] = "model-mtp.safetensors"

            (quant_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 0}, "weight_map": quant_wmap}, indent=2) + "\n"
            )
            _write_qwen4_exp_quant_config(
                quant_dir,
                raw_cfg,
                group_size=32,
                actorder="static",
                hadamard_block_size=32,
                quantize_mtp=True,
                quantize_vision=False,
                vision_fc2_channel=False,
                ple_group_size=16,
                ple_cd_passes=1,
            )

            # Verify packed tensor alignment
            _assert_packed_alignment(quant_dir)

            # Verify streaming forward on quantized checkpoint produces accurate logits & MTP outputs + routing rankings
            r_stats: dict[str, dict[str, torch.Tensor]] = {}
            q_final, q_pre_mixer, q_head = _stream_forward_qwen4_exp_states(
                quant_dir,
                cfg.text_config,
                calib_ids,
                device,
                return_pre_mixer=True,
                routing_stats_out=r_stats,
            )
            rankings = _write_hot_expert_rankings(quant_dir, r_stats, source_desc="unit_test")
            self.assertEqual(set(rankings.keys()), {"0", "1"})
            self.assertTrue((quant_dir / "runtime" / "static_hot_cache_rankings.json").exists())
            self.assertTrue((quant_dir / "runtime" / "expert_routing_profile.json").exists())
            q_logits_0 = q_head(q_final[0])
            cos_sim = F.cosine_similarity(
                ref_logits_0.float().flatten(), q_logits_0.float().flatten(), dim=0
            ).item()
            self.assertGreater(cos_sim, 0.96)

            q_mtp = _load_qwen4_exp_mtp_from_checkpoint(quant_dir, cfg.text_config, device)
            next_emb_0 = F.embedding(calib_ids[0][:, 1:], embed_w)
            ref_mtp_out = mtp(stream_pre_mixer[0][:, :-1, :], next_emb_0)
            q_mtp_out = q_mtp(q_pre_mixer[0][:, :-1, :], next_emb_0)
            mtp_cos = F.cosine_similarity(
                ref_mtp_out.float().flatten(), q_mtp_out.float().flatten(), dim=0
            ).item()
            self.assertGreater(mtp_cos, 0.95)

            # Verify PLE reuse works when grouped PLE files (1 file) < ngram_heads (2 heads)
            reuse_out_dir = Path(tmp) / "quant_reuse_ple"
            reuse_out_dir.mkdir(parents=True, exist_ok=True)
            layer1_reuse = _build_lightweight_decoder_layer(cfg.text_config, 1, device)
            _load_decoder_layer_weights(layer1_reuse, "model.language_model.layers.1", base_dir, wmap, device)
            h_calib_reuse = [
                inp.repeat(1, 1, cfg.text_config.hc_count)
                for inp in [F.embedding(ids, embed_w) for ids in calib_ids]
            ]
            wmap_without_ple = {k: v for k, v in wmap.items() if "ple_embedding" not in k}
            _, ple_wmap_reused, _, l1_reuse_counts = _quantize_decoder_layer_streaming(
                layer1_reuse,
                "model.language_model.layers.1",
                h_calib_reuse,
                pos_list,
                calib_ids,
                base_dir=base_dir,
                shard_map=wmap_without_ple,
                output_dir=reuse_out_dir,
                layer_idx=1,
                group_size=32,
                hadamard_block_size=32,
                actorder="static",
                dampening_frac=0.01,
                ple_group_size=16,
                ple_cd_passes=1,
                reuse_ple_dir=quant_dir,
            )
            self.assertEqual(l1_reuse_counts["ple_shards"], 2)
            self.assertTrue((reuse_out_dir / "model-ple-000.safetensors").exists())
            self.assertGreater(len(ple_wmap_reused), 0)

            # Verify ple_group_size=0 keeps Layer-1 PLE in BF16 without ZeroDivisionError
            bf16_ple_dir = Path(tmp) / "quant_bf16_ple"
            bf16_ple_dir.mkdir(parents=True, exist_ok=True)
            layer1_bf16 = _build_lightweight_decoder_layer(cfg.text_config, 1, device)
            _load_decoder_layer_weights(layer1_bf16, "model.language_model.layers.1", base_dir, wmap, device)
            h_calib_bf16 = [
                inp.repeat(1, 1, cfg.text_config.hc_count)
                for inp in [F.embedding(ids, embed_w) for ids in calib_ids]
            ]
            _, ple_wmap_bf16, _, l1_bf16_counts = _quantize_decoder_layer_streaming(
                layer1_bf16,
                "model.language_model.layers.1",
                h_calib_bf16,
                pos_list,
                calib_ids,
                base_dir=base_dir,
                shard_map=wmap,
                output_dir=bf16_ple_dir,
                layer_idx=1,
                group_size=32,
                hadamard_block_size=32,
                actorder="static",
                dampening_frac=0.01,
                ple_group_size=0,
                ple_cd_passes=1,
            )
            self.assertEqual(l1_bf16_counts["ple_shards"], 2)
            self.assertTrue(any(k.endswith(".weight") for k in ple_wmap_bf16))
            self.assertFalse(any(k.endswith(".weight_packed") for k in ple_wmap_bf16))
            _write_qwen4_exp_quant_config(
                bf16_ple_dir,
                raw_cfg,
                group_size=32,
                actorder="static",
                hadamard_block_size=32,
                quantize_mtp=False,
                quantize_vision=False,
                vision_fc2_channel=False,
                ple_group_size=0,
                ple_cd_passes=1,
            )
            bf16_qcfg = json.loads((bf16_ple_dir / "config.json").read_text())
            self.assertNotIn("ple_quantization", bf16_qcfg["quantization_config"])

    def test_config_regexes_and_expert_hadamard_hooks(self):
        import re

        from warp_quant.hadamard_gptq import (
            BASE_IGNORE,
            MTP_HAD_TARGET_RE,
            MTP_TARGET_RE,
            TEXT_HAD_TARGET_RE,
            TEXT_TARGET_RE,
        )
        from warp_quant.streaming_moe import _attach_expert_hadamard_hooks, _hadamard_rotate_last_dim

        self.assertEqual(TEXT_TARGET_RE, TEXT_HAD_TARGET_RE)
        self.assertEqual(MTP_TARGET_RE, MTP_HAD_TARGET_RE)
        text_pat = re.compile(TEXT_TARGET_RE.removeprefix("re:"))
        mtp_pat = re.compile(MTP_TARGET_RE.removeprefix("re:"))
        ignore_pats = [re.compile(p.removeprefix("re:")) for p in BASE_IGNORE if p.startswith("re:")]

        for mod_name in (
            "model.language_model.layers.0.mlp.experts.0.gate_proj",
            "model.language_model.layers.0.mlp.experts.511.up_proj",
            "model.language_model.layers.39.mlp.experts.12.down_proj",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj",
            "model.language_model.layers.0.mlp.shared_expert.up_proj",
            "model.language_model.layers.0.mlp.shared_expert.down_proj",
        ):
            self.assertIsNotNone(text_pat.match(mod_name), f"Expected TEXT_TARGET_RE to match {mod_name}")
            self.assertFalse(
                any(ip.match(mod_name) for ip in ignore_pats), f"Unexpected ignore match on {mod_name}"
            )

        for mtp_mod_name in (
            "mtp.layers.0.mlp.experts.0.gate_proj",
            "mtp.layers.0.mlp.experts.255.down_proj",
            "mtp.layers.0.mlp.shared_expert.gate_proj",
            "mtp.layers.0.mlp.shared_expert.up_proj",
            "mtp.layers.0.mlp.shared_expert.down_proj",
        ):
            self.assertIsNotNone(
                mtp_pat.match(mtp_mod_name), f"Expected MTP_TARGET_RE to match {mtp_mod_name}"
            )
            self.assertFalse(any(ip.match(mtp_mod_name) for ip in ignore_pats))

        for ignored_name in (
            "model.language_model.layers.0.mlp.shared_expert_gate",
            "mtp.layers.0.mlp.shared_expert_gate",
            "model.language_model.layers.0.mlp.gate",
        ):
            self.assertTrue(
                any(ip.match(ignored_name) for ip in ignore_pats), f"Expected ignore on {ignored_name}"
            )

        # Verify _attach_expert_hadamard_hooks + pre-rotated weights == unrotated experts (no quantization noise)
        torch.manual_seed(7)
        cfg = _make_tiny_qwen4_exp_config()
        layer_ref = _build_lightweight_decoder_layer(cfg.text_config, 0, torch.device("cpu"))
        layer_rot = _build_lightweight_decoder_layer(cfg.text_config, 0, torch.device("cpu"))
        for p_ref, p_rot in zip(layer_ref.parameters(), layer_rot.parameters(), strict=True):
            nn.init.normal_(p_ref, mean=0.0, std=0.05)
            p_rot.data.copy_(p_ref.data)

        exp_rot = layer_rot.mlp.experts
        exp_rot.gate_up_proj.data.copy_(
            _hadamard_rotate_last_dim(exp_rot.gate_up_proj.data, 32, compute_dtype=torch.float64)
        )
        exp_rot.down_proj.data.copy_(
            _hadamard_rotate_last_dim(exp_rot.down_proj.data, 32, compute_dtype=torch.float64)
        )
        _attach_expert_hadamard_hooks(exp_rot, 32, 32)

        x = torch.randn(10, cfg.text_config.hidden_size, dtype=torch.bfloat16)
        _, rw, sel = layer_ref.mlp.gate(x)
        out_ref = layer_ref.mlp.experts(x, sel, rw)
        out_rot = exp_rot(x, sel, rw)
        self.assertTrue(torch.allclose(out_ref, out_rot, atol=2e-3, rtol=1e-2))

    def test_qwen35_moe_streaming_quantize_and_mtp_roundtrip(self):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForConditionalGeneration,
            Qwen3_5MoeTextRotaryEmbedding,
        )

        from warp_quant.streaming_moe import Qwen35MoeMultiTokenPredictor

        torch.manual_seed(99)
        cfg = Qwen3_5MoeConfig(
            text_config={
                "vocab_size": 256,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [4, 2, 2],
                },
            }
        )
        _ensure_qwen4_exp_device_kernels(cfg.text_config, torch.device("cpu"))
        device = torch.device("cpu")
        model = Qwen3_5MoeForConditionalGeneration(cfg).to(torch.bfloat16).eval()
        mtp = Qwen35MoeMultiTokenPredictor(cfg.text_config, device)
        for p in mtp.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "base35"
            quant_dir = Path(tmp) / "quant35"
            base_dir.mkdir()
            quant_dir.mkdir()

            raw_cfg = cfg.to_dict()
            (base_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2) + "\n")

            base_tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
            for k, v in mtp.state_dict().items():
                base_tensors[f"mtp.{k}"] = v.contiguous()
            save_file(base_tensors, str(base_dir / "model-00001-of-00001.safetensors"))
            wmap = {k: "model-00001-of-00001.safetensors" for k in base_tensors}
            (base_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 0}, "weight_map": wmap}, indent=2) + "\n"
            )

            calib_ids = [
                torch.randint(3, 250, (1, 16), dtype=torch.long),
                torch.randint(3, 250, (1, 16), dtype=torch.long),
            ]
            ref_logits_0 = model(input_ids=calib_ids[0]).logits
            stream_final, _, stream_head = _stream_forward_qwen4_exp_states(
                base_dir, cfg.text_config, calib_ids, device
            )
            self.assertTrue(torch.allclose(ref_logits_0, stream_head(stream_final[0]), atol=2e-3, rtol=1e-2))

            rotary = Qwen3_5MoeTextRotaryEmbedding(cfg.text_config)
            embed_w = model.model.language_model.embed_tokens.weight.detach()
            h_calib = []
            pos_list = []
            for ids in calib_ids:
                inp = F.embedding(ids, embed_w)
                pids = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
                h_calib.append(inp)
                pos_list.append(rotary(inp, pids))

            s1_tensors = {
                k: v.clone()
                for k, v in base_tensors.items()
                if k.startswith(("model.language_model.embed_tokens.", "model.language_model.norm."))
            }
            s2_tensors = {"lm_head.weight": base_tensors["lm_head.weight"].clone()}

            rco_bank_dir = Path(tmp) / "rco_bank"
            rco_bank_dir.mkdir()
            teacher_qat_state = {
                "ids": [c[:, :16].cpu().clone() for c in calib_ids * 4],
                "h_list": [h[:, :16, :].cpu().clone() for h in h_calib * 4],
                "pos_emb_list": [(p[0][:, :, :16], p[1][:, :, :16]) for p in pos_list * 4],
            }

            for l_idx in range(2):
                layer = _build_lightweight_decoder_layer(cfg.text_config, l_idx, device)
                l_pref = f"model.language_model.layers.{l_idx}"
                _load_decoder_layer_weights(layer, l_pref, base_dir, wmap, device)
                l_tensors, _, _, l_counts = _quantize_decoder_layer_streaming(
                    layer,
                    l_pref,
                    h_calib,
                    pos_list,
                    calib_ids,
                    base_dir=base_dir,
                    shard_map=wmap,
                    output_dir=quant_dir,
                    layer_idx=l_idx,
                    group_size=32,
                    hadamard_block_size=32,
                    actorder="static",
                    dampening_frac=0.01,
                    rco_w2_bank_dir=rco_bank_dir,
                    teacher_qat_state=teacher_qat_state,
                )
                s1_tensors.update(l_tensors)
                self.assertEqual(l_counts["routed_experts"], 12)
                self.assertTrue((rco_bank_dir / f"layer_{l_idx:02d}_w2.pt").is_file())

            final_norm_t = model.model.language_model.norm
            with torch.no_grad():
                t_norm_list = [final_norm_t(th) for th in teacher_qat_state["h_list"]]
            torch.save(
                {"input_ids": teacher_qat_state["ids"], "final_norm_states": t_norm_list},
                rco_bank_dir / "qat_teacher_states.pt",
            )

            save_file(s1_tensors, str(quant_dir / "model-00001-of-00002.safetensors"))
            save_file(s2_tensors, str(quant_dir / "model-00002-of-00002.safetensors"))
            quant_wmap = {k: "model-00001-of-00002.safetensors" for k in s1_tensors}
            quant_wmap.update({k: "model-00002-of-00002.safetensors" for k in s2_tensors})

            (quant_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 0}, "weight_map": quant_wmap}, indent=2) + "\n"
            )
            _write_qwen4_exp_quant_config(
                quant_dir,
                raw_cfg,
                group_size=32,
                actorder="static",
                hadamard_block_size=32,
                quantize_mtp=False,
                quantize_vision=False,
                vision_fc2_channel=False,
            )
            q_cfg_written = json.loads((quant_dir / "config.json").read_text())
            self.assertNotIn("ple_quantization", q_cfg_written["quantization_config"])
            _assert_packed_alignment(quant_dir)

            r_stats: dict[str, dict[str, torch.Tensor]] = {}
            q_final, _, q_head = _stream_forward_qwen4_exp_states(
                quant_dir, cfg.text_config, calib_ids, device, routing_stats_out=r_stats
            )
            q_logits_0 = q_head(q_final[0])
            cos_sim = F.cosine_similarity(
                ref_logits_0.float().flatten(), q_logits_0.float().flatten(), dim=0
            ).item()
            self.assertGreater(cos_sim, 0.95)

            # Verify MoE RCO search + QAT + checkpoint assembly + streaming forward on mixed W2/W4 checkpoint
            from warp_quant.streaming_moe import run_moe_rco_search_and_qat

            rco_out_dir = Path(tmp) / "rco_out"
            rco_summary = run_moe_rco_search_and_qat(
                model_key="35b",
                base_dir=base_dir,
                w4_ckpt_dir=quant_dir,
                rco_w2_bank_dir=rco_bank_dir,
                out_root_dir=rco_out_dir,
                budgets=[0.5],
                group_size=32,
                hadamard_block_size=32,
                search_samples=4,
                steps=4,
                lr=0.1,
                scale_qat=True,
                qat_samples=4,
                qat_lr=1e-4,
                assemble_names=["50pct_rco"],
            )
            self.assertIn("50pct_rco", rco_summary["assembled"])
            rco_ckpt = rco_out_dir / "checkpoints" / "50pct_rco"
            self.assertTrue((rco_ckpt / "rco_manifest.json").is_file())
            self.assertTrue((rco_ckpt / "chat_template.jinja").is_file())
            rco_final, _, rco_head = _stream_forward_qwen4_exp_states(
                rco_ckpt, cfg.text_config, calib_ids, device
            )
            rco_logits_0 = rco_head(rco_final[0])
            self.assertEqual(rco_logits_0.shape, ref_logits_0.shape)
            self.assertTrue(torch.isfinite(rco_logits_0).all())

    def test_ple_mse_scale_search_hcd_and_kv_mmse_compensation(self):
        from warp_quant.streaming_moe import apply_ple_kv_mmse_compensation

        torch.manual_seed(42)
        d_h = 64
        u_dir = torch.randn(1, d_h)
        u_dir = u_dir / u_dir.norm()
        wk = torch.randn(128, d_h) + 12.0 * torch.randn(128, 1) @ u_dir
        wv = torch.randn(64, d_h) + 5.0 * torch.randn(64, 1) @ u_dir
        n_rows = 1024
        z = torch.randn(n_rows, d_h) * (1.0 + 0.35 * torch.randn(n_rows, d_h).abs())
        w_batch = z + 2.0 * (z @ u_dir.T) @ u_dir

        h_list, u_list = compute_ple_head_hessians(wk, wv, num_heads=1)
        h_m, u_m = h_list[0], u_list[0]

        q_legacy, _, _ = quantize_ple_batch_gptq_lut4(
            w_batch,
            h_m,
            u_m,
            group_size=32,
            cd_passes=1,
            mse_scale_search=False,
            refine_scales=False,
        )
        q_upgraded, _, _ = quantize_ple_batch_gptq_lut4(
            w_batch,
            h_m,
            u_m,
            group_size=32,
            cd_passes=1,
            mse_scale_search=True,
            refine_scales=True,
        )
        loss_legacy = float((((q_legacy - w_batch) @ h_m) * (q_legacy - w_batch)).sum().item())
        loss_upgraded = float((((q_upgraded - w_batch) @ h_m) * (q_upgraded - w_batch)).sum().item())
        self.assertLess(loss_upgraded, loss_legacy * 0.93)

        dummy_ple = nn.Module()
        dummy_ple.key_proj = nn.Linear(d_h, 128, bias=False)
        dummy_ple.value_proj = nn.Linear(d_h, 64, bias=False)
        dummy_ple.key_proj.weight.data.copy_(wk)
        dummy_ple.value_proj.weight.data.copy_(wv)

        k_ref = w_batch @ wk.T
        k_err_pre = (q_upgraded @ dummy_ple.key_proj.weight.T - k_ref).norm().item()
        g_m = (q_upgraded.T @ q_upgraded).to(torch.float64)
        c_m = ((w_batch - q_upgraded).T @ q_upgraded).to(torch.float64)
        comp_ratio = apply_ple_kv_mmse_compensation(dummy_ple, [g_m], [c_m], ridge_frac=0.01)
        self.assertGreater(comp_ratio, 0.0)
        k_err_post = (q_upgraded @ dummy_ple.key_proj.weight.T - k_ref).norm().item()
        self.assertLess(k_err_post, k_err_pre)

    def test_qwen4_exp_rco_search_qat_and_assembly(self):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpForConditionalGeneration,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextRotaryEmbedding,
        )

        from warp_quant.streaming_moe import run_moe_rco_search_and_qat

        torch.manual_seed(123)
        cfg = _make_tiny_qwen4_exp_config()
        device = torch.device("cpu")
        model = Qwen4ExpForConditionalGeneration(cfg).to(torch.bfloat16).eval()

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "base_q4"
            quant_dir = Path(tmp) / "quant_q4"
            rco_bank_dir = Path(tmp) / "rco_bank_q4"
            rco_out_dir = Path(tmp) / "rco_out_q4"
            base_dir.mkdir()
            quant_dir.mkdir()
            rco_bank_dir.mkdir()

            raw_cfg = cfg.to_dict()
            (base_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2) + "\n")

            base_tensors: dict[str, torch.Tensor] = {}
            for k, v in model.state_dict().items():
                if k.endswith("mlp.experts.gate_up_proj"):
                    pref = k[: -len("gate_up_proj")]
                    inter = v.shape[1] // 2
                    for e in range(v.shape[0]):
                        base_tensors[f"{pref}{e}.gate_proj.weight"] = v[e, :inter].contiguous()
                        base_tensors[f"{pref}{e}.up_proj.weight"] = v[e, inter:].contiguous()
                elif k.endswith("mlp.experts.down_proj"):
                    pref = k[: -len("down_proj")]
                    for e in range(v.shape[0]):
                        base_tensors[f"{pref}{e}.down_proj.weight"] = v[e].contiguous()
                elif k == "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight":
                    mid = v.shape[0] // 2
                    pref = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
                    base_tensors[f"{pref}.shard_0.weight"] = v[:mid].contiguous()
                    base_tensors[f"{pref}.shard_1.weight"] = v[mid:].contiguous()
                else:
                    base_tensors[k] = v.contiguous()

            save_file(base_tensors, str(base_dir / "model-00001-of-00001.safetensors"))
            wmap = {k: "model-00001-of-00001.safetensors" for k in base_tensors}
            (base_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 0}, "weight_map": wmap}, indent=2) + "\n"
            )

            calib_ids = [
                torch.randint(3, 250, (1, 16), dtype=torch.long),
                torch.randint(3, 250, (1, 16), dtype=torch.long),
            ]
            rotary = Qwen4ExpTextRotaryEmbedding(cfg.text_config)
            embed_w = model.model.language_model.embed_tokens.weight.detach()
            h_calib = []
            pos_list = []
            for ids in calib_ids:
                inp = F.embedding(ids, embed_w)
                pids = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
                h_calib.append(inp.repeat(1, 1, cfg.text_config.hc_count))
                pos_list.append(rotary(inp, pids))

            teacher_qat_state = {
                "input_ids": [c[:, :16].cpu().clone() for c in calib_ids * 4],
                "h_list": [h[:, :16, :].cpu().clone() for h in h_calib * 4],
                "pos_emb_list": [(p[0][:, :, :16], p[1][:, :, :16]) for p in pos_list * 4],
                "ple_embeds": {},
            }

            quant_wmap: dict[str, str] = {}
            embed_head = {
                k: v.clone()
                for k, v in base_tensors.items()
                if k.startswith(
                    (
                        "model.language_model.embed_tokens.",
                        "model.language_model.hyper_connection_mixer.",
                        "lm_head.",
                    )
                )
            }
            save_file(embed_head, str(quant_dir / "model-embed-head.safetensors"))
            for tk in embed_head:
                quant_wmap[tk] = "model-embed-head.safetensors"

            for l_idx in range(2):
                layer = _build_lightweight_decoder_layer(cfg.text_config, l_idx, device)
                l_pref = f"model.language_model.layers.{l_idx}"
                _load_decoder_layer_weights(layer, l_pref, base_dir, wmap, device)
                l_tensors, ple_wmap, _, _ = _quantize_decoder_layer_streaming(
                    layer,
                    l_pref,
                    h_calib,
                    pos_list,
                    calib_ids,
                    base_dir=base_dir,
                    shard_map=wmap,
                    output_dir=quant_dir,
                    layer_idx=l_idx,
                    group_size=32,
                    hadamard_block_size=32,
                    actorder="static",
                    dampening_frac=0.01,
                    ple_group_size=16,
                    ple_cd_passes=1,
                    rco_w2_bank_dir=rco_bank_dir,
                    teacher_qat_state=teacher_qat_state,
                )
                l_fname = f"model-layer-{l_idx:02d}.safetensors"
                save_file(l_tensors, str(quant_dir / l_fname))
                for tk in l_tensors:
                    quant_wmap[tk] = l_fname
                quant_wmap.update(ple_wmap)

            self.assertIn("1", teacher_qat_state["ple_embeds"])
            fn_mixer = Qwen4ExpTextGatedResidual(cfg.text_config, use_combine=False).to(torch.bfloat16).eval()
            fn_mixer.load_state_dict(
                {
                    k.removeprefix("model.language_model.hyper_connection_mixer."): v
                    for k, v in base_tensors.items()
                    if k.startswith("model.language_model.hyper_connection_mixer.")
                },
                strict=True,
            )
            with torch.no_grad():
                t_norm_list = [fn_mixer(th) for th in teacher_qat_state["h_list"]]
            torch.save(
                {
                    "input_ids": teacher_qat_state["input_ids"],
                    "final_norm_states": t_norm_list,
                    "ple_embeds": teacher_qat_state["ple_embeds"],
                },
                rco_bank_dir / "qat_teacher_states.pt",
            )

            (quant_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 0}, "weight_map": quant_wmap}, indent=2) + "\n"
            )
            _write_qwen4_exp_quant_config(
                quant_dir,
                raw_cfg,
                group_size=32,
                actorder="static",
                hadamard_block_size=32,
                quantize_mtp=False,
                quantize_vision=False,
                vision_fc2_channel=False,
                ple_group_size=16,
                ple_cd_passes=1,
            )

            rco_summary = run_moe_rco_search_and_qat(
                model_key="flash-next",
                base_dir=base_dir,
                w4_ckpt_dir=quant_dir,
                rco_w2_bank_dir=rco_bank_dir,
                out_root_dir=rco_out_dir,
                budgets=[0.5],
                group_size=32,
                hadamard_block_size=32,
                search_samples=4,
                steps=2,
                lr=0.1,
                scale_qat=True,
                qat_samples=4,
                qat_lr=1e-4,
                assemble_names=["50pct_rco", "all_high"],
            )
            self.assertIn("50pct_rco", rco_summary["assembled"])
            self.assertIn("all_high", rco_summary["assembled"])
            rco_ckpt = rco_out_dir / "checkpoints" / "50pct_rco"
            self.assertTrue((rco_ckpt / "model-layer-00.safetensors").is_file())
            self.assertTrue((rco_ckpt / "model-layer-01.safetensors").is_file())
            self.assertTrue((rco_ckpt / "model-ple-000.safetensors").is_file())
            rco_final, _, rco_head = _stream_forward_qwen4_exp_states(
                rco_ckpt, cfg.text_config, calib_ids, device
            )
            rco_logits = rco_head(rco_final[0])
            self.assertTrue(torch.isfinite(rco_logits).all())


if __name__ == "__main__":
    unittest.main()
