import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, json
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.utils.repro import set_global_seed
from src.run_experiment import run_experiment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["polynomial", "nonlinear", "healthcare", "csv"])
    p.add_argument("--mode", required=True,
                   choices=["convex_model", "convex_sample",
                            "nonconvex_model", "nonconvex_sample"])
    # CSV-only
    p.add_argument("--data_path")
    p.add_argument("--target_col")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--header", default="infer")
    # knobs
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--degrees", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    p.add_argument("--methods", nargs="*", default=["fixed", "halving", "diminishing"])
    p.add_argument("--fracs", type=float, nargs="*", default=[
        0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05,
        0.1, 0.15, 0.2, 0.25, 0.5, 1.0
    ])
    p.add_argument("--hidden_sizes", type=int, nargs="*", default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--width", type=int, default=50)
    p.add_argument("--degree", type=int, default=3)
    # seeds/out
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.outdir) / f"{args.dataset}_{args.mode}_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "manifest.json").write_text(json.dumps(vars(args), indent=2))

    all_records = []

    for seed in args.seeds:
        print(f"\n=== Running seed {seed} ===")
        set_global_seed(seed)
        seed_dir = run_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        recs = run_experiment(
            dataset=args.dataset,
            mode=args.mode,
            seed=seed,
            save_dir=str(seed_dir),
            epochs=args.epochs,
            degrees=args.degrees,
            methods=args.methods,
            fracs=args.fracs,
            hidden_sizes=args.hidden_sizes,
            width=args.width,
            degree=args.degree,
            data_path=args.data_path,
            target_col=args.target_col,
            delimiter=args.delimiter,
            header=args.header,
        )
        all_records.extend(recs)

    # one compiled csv
    if all_records:
        df = pd.DataFrame(all_records)
        family_dir = Path(args.outdir) / f"{args.dataset}_{args.mode}"
        family_dir.mkdir(parents=True, exist_ok=True)
        family_csv = family_dir / "all_records.csv"
        df.to_csv(family_csv, index=False)  
        print(f"\n📦 Saved {len(df)} records to {family_csv}")


if __name__ == "__main__":
    main()
