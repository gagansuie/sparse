#!/usr/bin/env python
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
TENPAK_BIN = Path(
    os.environ.get("TENPAK_BIN", str(ROOT / "target" / "release" / "tenpak"))
)
README_PATH = Path(os.environ.get("TENPAK_README_PATH", str(ROOT / "README.md")))

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

    size_int8_bytes = base_int8_artifact.stat().st_size
    size_int4_bytes = base_int4_artifact.stat().st_size
    size_int8_gb = size_int8_bytes / 1e9
    size_int4_gb = size_int4_bytes / 1e9

    # Decompress artifacts back to JSON bundles
    base_int8_bundle_path = TMP_DIR / "base_bundle_int8.json"
    base_int4_bundle_path = TMP_DIR / "base_bundle_int4.json"

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

    with base_int8_bundle_path.open() as f:
        base_int8_bundle = json.load(f)
    with base_int4_bundle_path.open() as f:
        base_int4_bundle = json.load(f)

    # Rebuild models from quantized bundles and compute perplexity
    print("[tenpak] Evaluating int8 reconstructed model...")
    int8_sd = bundle_to_state_dict(base_int8_bundle)
    model_int8 = AutoModelForCausalLM.from_pretrained(str(LOCAL_MODEL_DIR))
    model_int8.load_state_dict(int8_sd)
    model_int8.to(device)
    ppl_int8 = compute_perplexity(model_int8, tokenizer, texts, device)

    print("[tenpak] Evaluating int4 reconstructed model...")
    int4_sd = bundle_to_state_dict(base_int4_bundle)
    model_int4 = AutoModelForCausalLM.from_pretrained(str(LOCAL_MODEL_DIR))
    model_int4.load_state_dict(int4_sd)
    model_int4.to(device)
    ppl_int4 = compute_perplexity(model_int4, tokenizer, texts, device)

    compression_int8 = size_fp_bytes / size_int8_bytes if size_int8_bytes > 0 else float("nan")
    compression_int4 = size_fp_bytes / size_int4_bytes if size_int4_bytes > 0 else float("nan")
    delta_ppl_int8 = ppl_int8 - ppl_fp
    delta_ppl_int4 = ppl_int4 - ppl_fp

    print("[tenpak] Codec results:")
    print(f"  FP baseline : size={size_fp_gb:.3f} GB, ppl={ppl_fp:.4f}")
    print(f"  tenpak int8 : size={size_int8_gb:.3f} GB, ratio={compression_int8:.2f}x, ppl={ppl_int8:.4f} (Δ={delta_ppl_int8:+.4f})")
    print(f"  tenpak int4 : size={size_int4_gb:.3f} GB, ratio={compression_int4:.2f}x, ppl={ppl_int4:.4f} (Δ={delta_ppl_int4:+.4f})")

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

    # Update README tables
    readme = README_PATH.read_text()

    old_codec_block = (
        "| FP baseline       |                   | 1.0×              |                |              |               |           |\n"
        "| tenpak int8        |                   |                   |                |              |               |           |\n"
        "| tenpak int4        |                   |                   |                |              |               |           |"
    )

    new_codec_block = (
        f"| FP baseline       | {size_fp_gb:.3f} | 1.0×              | {ppl_fp:.3f} | 0.0          | N/A           | N/A       |\n"
        f"| tenpak int8        | {size_int8_gb:.3f} | {compression_int8:.2f}× | {ppl_int8:.3f} | {delta_ppl_int8:+.3f} | N/A           | N/A       |\n"
        f"| tenpak int4        | {size_int4_gb:.3f} | {compression_int4:.2f}× | {ppl_int4:.3f} | {delta_ppl_int4:+.3f} | N/A           | N/A       |"
    )

    readme = readme.replace(old_codec_block, new_codec_block)

    old_delta_block = (
        "| Full FP fine-tune           | `base_fp.pt` + `ft_fp.pt`           | `S_base_fp + S_ft_fp`          | Two full-precision checkpoints.           |\n"
        "| Full tenpak fine-tune        | `base_fp.pt` + `ft_int4.tenpak`  | `S_base_fp + S_ft_int4`        | Compress the fine-tune only.              |\n"
        "| tenpak base + delta (A)      | `base_int4.tenpak` + `ft_delta`  | `S_base_int4 + S_delta`        | Compressed base + small variant delta.    |"
    )

    new_delta_block = (
        f"| Full FP fine-tune           | `base_fp.pt` + `ft_fp.pt`           | {total_full_fp_gb:.3f}         | Two full-precision checkpoints.           |\n"
        f"| Full tenpak fine-tune        | `base_fp.pt` + `ft_int4.tenpak`  | {total_full_int4_gb:.3f}       | Compress the fine-tune only.              |\n"
        f"| tenpak base + delta (A)      | `base_int4.tenpak` + `ft_delta`  | {total_base_delta_gb:.3f}      | Compressed base + small variant delta.    |"
    )

    readme = readme.replace(old_delta_block, new_delta_block)

    # Print the updated README to stdout instead of writing to file
    print("\n" + "="*80)
    print("[tenpak] UPDATED README CONTENT - Copy and paste this into your README.md:")
    print("="*80 + "\n")
    print(readme)
    print("\n" + "="*80)
    print(f"[tenpak] Evaluation completed for model {MODEL_NAME}.")
    print("[tenpak] Copy the content above and paste it into your README.md file.")
    print("="*80)


if __name__ == "__main__":
    main()
