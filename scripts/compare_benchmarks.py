import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = ROOT / "scripts" / "benchmark_decode.py"


def run_case(env_prefix, name, impl, extra_args):
    cmd = env_prefix + [
        "python",
        str(BENCH_SCRIPT),
        "--impl",
        impl,
        *extra_args,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    metrics = {"name": name, "impl": impl, "stdout": proc.stdout.strip()}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-conda-run", action="store_true")
    parser.add_argument("--conda-env", type=str, default="ai_infra_env_new")
    args = parser.parse_args()

    env_prefix = []
    if args.use_conda_run:
        env_prefix = ["conda", "run", "-n", args.conda_env]

    common = [
        "--emb-dim", "64",
        "--hidden-dim", "128",
        "--n-heads", "4",
        "--n-layers", "2",
        "--context-length", "32",
        "--prompt-length", "8",
        "--max-new-tokens", "16",
        "--trials", "3",
        "--vocab-size", "512",
    ]

    cases = [
        ("dense_nocache", "naive", common + ["--layer-ids"]),
        ("dense_nocache", "optimized", common + ["--layer-ids"]),
        ("dense_cache", "optimized", common + ["--layer-ids", "--use-cache"]),
        (
            "engram_nocache",
            "naive",
            common
            + [
                "--layer-ids", "0",
                "--engram-vocab-size", "257", "263",
                "--n-embed-per-ngram", "32",
                "--n-head-per-ngram", "4",
                "--kernel-size", "2",
            ],
        ),
        (
            "engram_nocache",
            "optimized",
            common
            + [
                "--layer-ids", "0",
                "--engram-vocab-size", "257", "263",
                "--n-embed-per-ngram", "32",
                "--n-head-per-ngram", "4",
                "--kernel-size", "2",
            ],
        ),
        (
            "engram_cache",
            "optimized",
            common
            + [
                "--layer-ids", "0",
                "--engram-vocab-size", "257", "263",
                "--n-embed-per-ngram", "32",
                "--n-head-per-ngram", "4",
                "--kernel-size", "2",
                "--use-cache",
            ],
        ),
        (
            "mhc_dense_nocache",
            "optimized",
            common + ["--layer-ids", "--hc-mult", "3"],
        ),
    ]

    results = [run_case(env_prefix, name, impl, extra) for name, impl, extra in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
