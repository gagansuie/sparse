#!/usr/bin/env python
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
TENPAK_BIN = Path(
    os.environ.get("TENPAK_BIN", str(ROOT / "target" / "release" / "tenpak"))
)

MODEL_NAME = os.environ.get("TENPAK_EVAL_MODEL", "gpt2")
SAFE_NAME = MODEL_NAME.replace("/", "_")
LOCAL_MODEL_DIR = MODELS_DIR / SAFE_NAME
FT_FP_DIR = MODELS_DIR / f"{SAFE_NAME}_ft_fp"
TMP_DIR = ROOT / "tmp_eval"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def ensure_model():
    print(f"[tenpak] Using model {MODEL_NAME}")
    if LOCAL_MODEL_DIR.exists():
        print(f"[tenpak] Loading model from {LOCAL_MODEL_DIR}")
        tok = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
        model = AutoModelForCausalLM.from_pretrained(str(LOCAL_MODEL_DIR))
    else:
        print(f"[tenpak] Local dir {LOCAL_MODEL_DIR} not found, downloading from hub")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(LOCAL_MODEL_DIR))
        model.save_pretrained(str(LOCAL_MODEL_DIR))
    return model, tok


def load_eval_dataset(num_examples: int = 128):
    print("[tenpak] Loading Wikitext-2 test split")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if num_examples and num_examples < len(ds):
        ds = ds.select(range(num_examples))
    texts = [x["text"] for x in ds if x["text"].strip()]
    return texts


def compute_perplexity(model, tokenizer, texts, device, max_length: int = 512) -> float:
    model.eval()
    nll = 0.0
    ntokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            n_tokens = input_ids.numel()
            nll += loss * n_tokens
            ntokens += n_tokens
    if ntokens == 0:
        return float("nan")
    return math.exp(nll / ntokens)


def state_dict_to_bundle(sd) -> dict:
    tensors = []
    for name, tensor in sd.items():
        t = tensor.detach().cpu().float()
        shape = list(t.shape)
        data = t.reshape(-1).tolist()
        tensors.append({"name": name, "shape": shape, "data": data})
    return {"tensors": tensors}


def bundle_to_state_dict(bundle: dict):
    sd = {}
    for t in bundle["tensors"]:
        name = t["name"]
        shape = t["shape"]
        data = torch.tensor(t["data"], dtype=torch.float32).view(*shape)
        sd[name] = data
    return sd


def evaluate_bundle_with_model(label, bundle, model, tokenizer, texts, device):
    """Load bundle weights into `model`, run perplexity, then free GPU memory."""
    print(f"[tenpak] Evaluating {label}...")
    sd = bundle_to_state_dict(bundle)
    model.load_state_dict(sd)
    model.to(device)
    ppl = compute_perplexity(model, tokenizer, texts, device)
    model.to("cpu")
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.empty_cache()
    del sd
    return ppl


def collect_activation_stats(
    model,
    tokenizer,
    texts,
    device,
    max_batches: int = 32,
    max_length: int = 256,
):
    print(
        f"[tenpak] Collecting activation stats for AWQ calibration (samples={max_batches})..."
    )
    model.eval()
    activation_sums = {}
    activation_counts = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, _output):
            if not inputs:
                return
            act = inputs[0].detach()
            if act.dim() == 1:
                act = act.unsqueeze(0)
            elif act.dim() > 2:
                act = act.view(-1, act.shape[-1])
            if act.numel() == 0:
                return
            mean_abs = act.abs().mean(dim=0).cpu()
            if name not in activation_sums:
                activation_sums[name] = mean_abs
                activation_counts[name] = 1
            else:
                activation_sums[name] += mean_abs
                activation_counts[name] += 1

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))

    calib_count = min(max_batches, len(texts))
    for text in texts[:calib_count]:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids)

    for handle in handles:
        handle.remove()

    stats = {}
    for name, tensor_sum in activation_sums.items():
        count = activation_counts.get(name, 0)
        if count == 0:
            continue
        mean = (tensor_sum / count).tolist()
        stats[f"{name}.weight"] = {"activation_means": mean}

    print(
        f"[tenpak] Captured activation stats for {len(stats)} tensors using {calib_count} samples."
    )
    return stats


def run_tenpak_cli(args):
    cmd = [str(TENPAK_BIN)] + args
    print(f"[tenpak] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    if not TENPAK_BIN.exists():
        raise SystemExit(
            f"tenpak binary not found at {TENPAK_BIN}. Run 'cargo build --release' first."
        )

    model, tokenizer = ensure_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tenpak] Using device: {device}")
    model.to(device)

    texts = load_eval_dataset(num_examples=int(os.environ.get("TENPAK_EVAL_SAMPLES", "128")))

    # Baseline FP metrics
    print("[tenpak] Computing baseline perplexity...")
    t0 = time.time()
    ppl_fp = compute_perplexity(model, tokenizer, texts, device)
    t1 = time.time()
    size_fp_bytes = dir_size_bytes(LOCAL_MODEL_DIR)
    size_fp_gb = size_fp_bytes / 1e9
    print(f"[tenpak] Baseline perplexity: {ppl_fp:.4f}, size={size_fp_gb:.3f} GB, time={t1-t0:.1f}s")

    # Build bundle and compress with int8/int4
    print("[tenpak] Building bundle from state_dict (this may take a while)...")
    base_sd = model.state_dict()
    base_bundle = state_dict_to_bundle(base_sd)
    base_bundle_path = TMP_DIR / "base_bundle.json"
    with base_bundle_path.open("w") as f:
        json.dump(base_bundle, f)

    # Sanity-check: rebuild model directly from the float bundle to ensure
    # state_dict⇄bundle conversion preserves perplexity before quantization.
    # Move baseline model off GPU to reclaim memory before further evals
    model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    del model

    print("[tenpak] Verifying FP bundle round-trip...")
    eval_model = AutoModelForCausalLM.from_pretrained(str(LOCAL_MODEL_DIR))
    ppl_bundle = evaluate_bundle_with_model(
        "FP bundle round-trip",
        base_bundle,
        eval_model,
        tokenizer,
        texts,
        device,
    )
    delta_bundle = ppl_bundle - ppl_fp
    print(
        f"[tenpak] Bundle round-trip perplexity: {ppl_bundle:.4f} (Δ {delta_bundle:+.4f} vs FP)"
    )
    eval_model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    base_int8_artifact = TMP_DIR / "base_int8.tenpak"
    base_int4_artifact = TMP_DIR / "base_int4.tenpak"

    run_tenpak_cli([
        "compress",
        "--input",
        str(base_bundle_path),
        "--output",
        str(base_int8_artifact),
        "--codec",
        "int8_sym_v1",
    ])
    run_tenpak_cli([
        "compress",
        "--input",
        str(base_bundle_path),
        "--output",
        str(base_int4_artifact),
        "--codec",
        "int4_sym_v1",
    ])
    
    base_int4_perchannel_artifact = TMP_DIR / "base_int4_perchannel.tenpak"
    run_tenpak_cli([
        "compress",
        "--input",
        str(base_bundle_path),
        "--output",
        str(base_int4_perchannel_artifact),
        "--codec",
        "int4_perchannel_v1",
    ])
    
    base_int4_sparse_artifact = TMP_DIR / "base_int4_sparse.tenpak"
    run_tenpak_cli([
        "compress",
        "--input",
        str(base_bundle_path),
        "--output",
        str(base_int4_sparse_artifact),
        "--codec",
        "int4_perchannel_sparse50_v1",
    ])
    
    base_int2_artifact = TMP_DIR / "base_int2.tenpak"
    run_tenpak_cli([
        "compress",
        "--input",
        str(base_bundle_path),
        "--output",
        str(base_int2_artifact),
        "--codec",
        "int2_sym_v1",
    ])
    
    base_int4_awq_artifact = TMP_DIR / "base_int4_awq.tenpak"
    run_tenpak_cli([
        "compress",
        "--input",
        str(base_bundle_path),
        "--output",
        str(base_int4_awq_artifact),
        "--codec",
        "int4_awq_v1",
    ])
    
    base_int4_g128_artifact = TMP_DIR / "base_int4_g128.tenpak"
    run_tenpak_cli([
        "compress",
        "--input",
        str(base_bundle_path),
        "--output",
        str(base_int4_g128_artifact),
        "--codec",
        "int4_g128_v1",
    ])

    size_int8_bytes = base_int8_artifact.stat().st_size
    size_int4_bytes = base_int4_artifact.stat().st_size
    size_int4_perchannel_bytes = base_int4_perchannel_artifact.stat().st_size
    size_int4_sparse_bytes = base_int4_sparse_artifact.stat().st_size
    size_int2_bytes = base_int2_artifact.stat().st_size
    size_int4_awq_bytes = base_int4_awq_artifact.stat().st_size
    size_int4_g128_bytes = base_int4_g128_artifact.stat().st_size
    size_int8_gb = size_int8_bytes / 1e9
    size_int4_gb = size_int4_bytes / 1e9
    size_int4_perchannel_gb = size_int4_perchannel_bytes / 1e9
    size_int4_sparse_gb = size_int4_sparse_bytes / 1e9
    size_int2_gb = size_int2_bytes / 1e9
    size_int4_awq_gb = size_int4_awq_bytes / 1e9
    size_int4_g128_gb = size_int4_g128_bytes / 1e9

    # Decompress artifacts back to JSON bundles
    base_int8_bundle_path = TMP_DIR / "base_bundle_int8.json"
    base_int4_bundle_path = TMP_DIR / "base_bundle_int4.json"
    base_int4_perchannel_bundle_path = TMP_DIR / "base_bundle_int4_perchannel.json"
    base_int4_sparse_bundle_path = TMP_DIR / "base_bundle_int4_sparse.json"
    base_int2_bundle_path = TMP_DIR / "base_bundle_int2.json"
    base_int4_awq_bundle_path = TMP_DIR / "base_bundle_int4_awq.json"
    base_int4_g128_bundle_path = TMP_DIR / "base_bundle_int4_g128.json"

    run_tenpak_cli([
        "decompress",
        "--input",
        str(base_int8_artifact),
        "--output",
        str(base_int8_bundle_path),
    ])
    run_tenpak_cli([
        "decompress",
        "--input",
        str(base_int4_artifact),
        "--output",
        str(base_int4_bundle_path),
    ])
    run_tenpak_cli([
        "decompress",
        "--input",
        str(base_int4_perchannel_artifact),
        "--output",
        str(base_int4_perchannel_bundle_path),
    ])
    run_tenpak_cli([
        "decompress",
        "--input",
        str(base_int4_sparse_artifact),
        "--output",
        str(base_int4_sparse_bundle_path),
    ])
    run_tenpak_cli([
        "decompress",
        "--input",
        str(base_int2_artifact),
        "--output",
        str(base_int2_bundle_path),
    ])
    run_tenpak_cli([
        "decompress",
        "--input",
        str(base_int4_awq_artifact),
        "--output",
        str(base_int4_awq_bundle_path),
    ])
    run_tenpak_cli([
        "decompress",
        "--input",
        str(base_int4_g128_artifact),
        "--output",
        str(base_int4_g128_bundle_path),
    ])

    with base_int8_bundle_path.open() as f:
        base_int8_bundle = json.load(f)
    with base_int4_bundle_path.open() as f:
        base_int4_bundle = json.load(f)
    with base_int4_perchannel_bundle_path.open() as f:
        base_int4_perchannel_bundle = json.load(f)
    with base_int4_sparse_bundle_path.open() as f:
        base_int4_sparse_bundle = json.load(f)
    with base_int2_bundle_path.open() as f:
        base_int2_bundle = json.load(f)
    with base_int4_awq_bundle_path.open() as f:
        base_int4_awq_bundle = json.load(f)
    with base_int4_g128_bundle_path.open() as f:
        base_int4_g128_bundle = json.load(f)

    # Rebuild models from quantized bundles and compute perplexity
    eval_model = AutoModelForCausalLM.from_pretrained(str(LOCAL_MODEL_DIR))
    ppl_int8 = evaluate_bundle_with_model(
        "int8 reconstructed model",
        base_int8_bundle,
        eval_model,
        tokenizer,
        texts,
        device,
    )
    ppl_int4 = evaluate_bundle_with_model(
        "int4 (per-tensor) reconstructed model",
        base_int4_bundle,
        eval_model,
        tokenizer,
        texts,
        device,
    )
    ppl_int4_perchannel = evaluate_bundle_with_model(
        "int4 (per-channel) reconstructed model",
        base_int4_perchannel_bundle,
        eval_model,
        tokenizer,
        texts,
        device,
    )
    ppl_int4_sparse = evaluate_bundle_with_model(
        "int4 (sparse 50%) reconstructed model",
        base_int4_sparse_bundle,
        eval_model,
        tokenizer,
        texts,
        device,
    )
    ppl_int2 = evaluate_bundle_with_model(
        "int2 reconstructed model",
        base_int2_bundle,
        eval_model,
        tokenizer,
        texts,
        device,
    )
    ppl_int4_awq = evaluate_bundle_with_model(
        "int4 AWQ reconstructed model",
        base_int4_awq_bundle,
        eval_model,
        tokenizer,
        texts,
        device,
    )
    ppl_int4_g128 = evaluate_bundle_with_model(
        "int4 g128 reconstructed model",
        base_int4_g128_bundle,
        eval_model,
        tokenizer,
        texts,
        device,
    )
    del eval_model

    compression_int8 = size_fp_bytes / size_int8_bytes if size_int8_bytes > 0 else float("nan")
    compression_int4 = size_fp_bytes / size_int4_bytes if size_int4_bytes > 0 else float("nan")
    compression_int4_perchannel = size_fp_bytes / size_int4_perchannel_bytes if size_int4_perchannel_bytes > 0 else float("nan")
    compression_int4_sparse = size_fp_bytes / size_int4_sparse_bytes if size_int4_sparse_bytes > 0 else float("nan")
    compression_int2 = size_fp_bytes / size_int2_bytes if size_int2_bytes > 0 else float("nan")
    compression_int4_awq = size_fp_bytes / size_int4_awq_bytes if size_int4_awq_bytes > 0 else float("nan")
    compression_int4_g128 = size_fp_bytes / size_int4_g128_bytes if size_int4_g128_bytes > 0 else float("nan")
    delta_ppl_int8 = ppl_int8 - ppl_fp
    delta_ppl_int4 = ppl_int4 - ppl_fp
    delta_ppl_int4_perchannel = ppl_int4_perchannel - ppl_fp
    delta_ppl_int4_sparse = ppl_int4_sparse - ppl_fp
    delta_ppl_int2 = ppl_int2 - ppl_fp
    delta_ppl_int4_awq = ppl_int4_awq - ppl_fp
    delta_ppl_int4_g128 = ppl_int4_g128 - ppl_fp

    print("[tenpak] Codec results:")
    print(f"  FP baseline          : size={size_fp_gb:.3f} GB, ppl={ppl_fp:.4f}")
    print(f"  tenpak int8          : size={size_int8_gb:.3f} GB, ratio={compression_int8:.2f}x, ppl={ppl_int8:.4f} (Δ={delta_ppl_int8:+.4f})")
    print(f"  tenpak int4 (tensor) : size={size_int4_gb:.3f} GB, ratio={compression_int4:.2f}x, ppl={ppl_int4:.4f} (Δ={delta_ppl_int4:+.4f})")
    print(f"  tenpak int4 (channel): size={size_int4_perchannel_gb:.3f} GB, ratio={compression_int4_perchannel:.2f}x, ppl={ppl_int4_perchannel:.4f} (Δ={delta_ppl_int4_perchannel:+.4f})")
    print(f"  tenpak int4 (sparse) : size={size_int4_sparse_gb:.3f} GB, ratio={compression_int4_sparse:.2f}x, ppl={ppl_int4_sparse:.4f} (Δ={delta_ppl_int4_sparse:+.4f})")
    print(f"  tenpak int2          : size={size_int2_gb:.3f} GB, ratio={compression_int2:.2f}x, ppl={ppl_int2:.4f} (Δ={delta_ppl_int2:+.4f})")
    print(f"  tenpak int4 AWQ      : size={size_int4_awq_gb:.3f} GB, ratio={compression_int4_awq:.2f}x, ppl={ppl_int4_awq:.4f} (Δ={delta_ppl_int4_awq:+.4f})")
    print(f"  tenpak int4 g128     : size={size_int4_g128_gb:.3f} GB, ratio={compression_int4_g128:.2f}x, ppl={ppl_int4_g128:.4f} (Δ={delta_ppl_int4_g128:+.4f})")
    
    # AWQ baseline comparison (from README)
    ppl_pct_delta_awq = (delta_ppl_int4_awq / ppl_fp) * 100 if ppl_fp > 0 else float("nan")
    ppl_pct_delta_g128 = (delta_ppl_int4_g128 / ppl_fp) * 100 if ppl_fp > 0 else float("nan")
    print(f"\n[tenpak] AWQ comparison:")
    print(f"  int4 AWQ % delta     : {ppl_pct_delta_awq:+.2f}% (target: <1%)")
    print(f"  int4 g128 % delta    : {ppl_pct_delta_g128:+.2f}% (target: <1%)")
    if abs(ppl_pct_delta_awq) < 1.0:
        print(f"  int4 AWQ meets <1% target!")
    if abs(ppl_pct_delta_g128) < 1.0:
        print(f"  int4 g128 meets <1% target!")

    # Simulated small fine-tune: modify a small subset of parameters
    print(f"[tenpak] Creating simulated fine-tune by modifying a subset of weights...")
    ft_sd = {}
    all_keys = list(base_sd.keys())
    n_change = max(1, len(all_keys) // 20)  # change ~5% of tensors
    change_keys = set(all_keys[-n_change:])

    for name, tensor in base_sd.items():
        if name in change_keys:
            noise = torch.randn_like(tensor) * 0.01
            ft_sd[name] = tensor + noise
        else:
            ft_sd[name] = tensor.clone()

    # Save full-precision fine-tune checkpoint
    model_ft = AutoModelForCausalLM.from_pretrained(str(LOCAL_MODEL_DIR))
    model_ft.load_state_dict(ft_sd)
    FT_FP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[tenpak] Saving simulated fine-tune to {FT_FP_DIR}")
    model_ft.save_pretrained(str(FT_FP_DIR))

    size_base_fp_gb = size_fp_gb
    size_ft_fp_bytes = dir_size_bytes(FT_FP_DIR)
    size_ft_fp_gb = size_ft_fp_bytes / 1e9

    # Build fine-tune bundle and compress
    print("[tenpak] Building fine-tune bundle and compressing with int4...")
    ft_bundle = state_dict_to_bundle(ft_sd)
    ft_bundle_path = TMP_DIR / "ft_bundle.json"
    with ft_bundle_path.open("w") as f:
        json.dump(ft_bundle, f)

    ft_int4_artifact = TMP_DIR / "ft_int4.tenpak"
    run_tenpak_cli([
        "compress",
        "--input",
        str(ft_bundle_path),
        "--output",
        str(ft_int4_artifact),
        "--codec",
        "int4_sym_v1",
    ])

    size_ft_int4_bytes = ft_int4_artifact.stat().st_size
    size_ft_int4_gb = size_ft_int4_bytes / 1e9
    size_base_int4_gb = size_int4_gb

    # Base + delta artifact
    ft_delta_artifact = TMP_DIR / "ft_delta.tenpak"
    run_tenpak_cli([
        "delta",
        "--base",
        str(base_int4_artifact),
        "--variant",
        str(ft_bundle_path),
        "--output",
        str(ft_delta_artifact),
        "--epsilon",
        "0.001",
    ])
    size_delta_bytes = ft_delta_artifact.stat().st_size
    size_delta_gb = size_delta_bytes / 1e9

    total_full_fp_gb = size_base_fp_gb + size_ft_fp_gb
    total_full_int4_gb = size_base_fp_gb + size_ft_int4_gb
    total_base_delta_gb = size_base_int4_gb + size_delta_gb

    print("[tenpak] Fine-tune storage results:")
    print(f"  Full FP       : {total_full_fp_gb:.3f} GB (base + FT)")
    print(f"  Full int4 FT  : {total_full_int4_gb:.3f} GB (base FP + FT int4)")
    print(f"  Base+Delta    : {total_base_delta_gb:.3f} GB (base int4 + delta)")

    # Build Markdown snippets for results (without editing README)
    codec_table = (
        "| Variant                     | On-disk size (GB) | Compression vs FP | Perplexity | Δ Perplexity |\n"
        "|-----------------------------|-------------------|-------------------|------------|--------------|\n"
        f"| FP baseline                 | {size_fp_gb:.3f} | 1.0×              | {ppl_fp:.3f} | 0.0          |\n"
        f"| tenpak int8                 | {size_int8_gb:.3f} | {compression_int8:.2f}× | {ppl_int8:.3f} | {delta_ppl_int8:+.3f} |\n"
        f"| tenpak int4 (tensor)        | {size_int4_gb:.3f} | {compression_int4:.2f}× | {ppl_int4:.3f} | {delta_ppl_int4:+.3f} |\n"
        f"| tenpak int4 (channel)       | {size_int4_perchannel_gb:.3f} | {compression_int4_perchannel:.2f}× | {ppl_int4_perchannel:.3f} | {delta_ppl_int4_perchannel:+.3f} |\n"
        f"| tenpak int4 (sparse 50%)    | {size_int4_sparse_gb:.3f} | {compression_int4_sparse:.2f}× | {ppl_int4_sparse:.3f} | {delta_ppl_int4_sparse:+.3f} |\n"
        f"| tenpak int2                 | {size_int2_gb:.3f} | {compression_int2:.2f}× | {ppl_int2:.3f} | {delta_ppl_int2:+.3f} |\n"
        f"| tenpak int4 AWQ             | {size_int4_awq_gb:.3f} | {compression_int4_awq:.2f}× | {ppl_int4_awq:.3f} | {delta_ppl_int4_awq:+.3f} |\n"
        f"| tenpak int4 g128            | {size_int4_g128_gb:.3f} | {compression_int4_g128:.2f}× | {ppl_int4_g128:.3f} | {delta_ppl_int4_g128:+.3f} |\n"
    )

    storage_table = (
        "| Variant                     | Files stored                         | Total on-disk size (GB) | Notes                                      |\n"
        "|-----------------------------|--------------------------------------|--------------------------|--------------------------------------------|\n"
        f"| Full FP fine-tune           | `base_fp.pt` + `ft_fp.pt`           | {total_full_fp_gb:.3f}   | Two full-precision checkpoints.           |\n"
        f"| Full tenpak fine-tune       | `base_fp.pt` + `ft_int4.tenpak`    | {total_full_int4_gb:.3f} | Compress the fine-tune only.              |\n"
        f"| tenpak base + delta (A)     | `base_int4.tenpak` + `ft_delta`    | {total_base_delta_gb:.3f}| Compressed base + small variant delta.    |\n"
    )

    # Print just the results block for manual copy/paste
    print("\n" + "="*80)
    print("[tenpak] EVALUATION RESULTS (Markdown snippet)")
    print("="*80 + "\n")
    print(f"Model: {MODEL_NAME}")
    print("\n### Codec vs. quality\n")
    print(codec_table)
    print("\n### Base + delta fine-tune storage\n")
    print(storage_table)
    print("\n" + "="*80)
    print(f"[tenpak] Evaluation completed for model {MODEL_NAME}.")
    print("[tenpak] Copy/paste the snippet above wherever you track results (README, Notion, etc.).")
    print("="*80)


if __name__ == "__main__":
    main()
