import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--output", default="sparse_demo_deploy_out")
    parser.add_argument("--engine", default="tgi")
    parser.add_argument("--hardware", default="cpu")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--eval-samples", type=int, default=3)
    parser.add_argument("--benchmark-samples", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        "-m",
        "cli.main",
        "deploy",
        args.model_id,
        "--engine",
        args.engine,
        "--hardware",
        args.hardware,
        "--max-ppl-delta",
        "100.0",
        "--max-latency",
        "1000000000",
        "--min-throughput",
        "0",
        "--eval-samples",
        str(args.eval_samples),
        "--benchmark-samples",
        str(args.benchmark_samples),
        "--no-calibration",
        "--candidates",
        "fp16",
        "int4_residual",
        "--output",
        args.output,
    ]

    if args.backend is not None:
        cmd.extend(["--backend", args.backend])

    print("Running:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
