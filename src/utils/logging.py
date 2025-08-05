import os
import json
import pandas as pd

def save_results(log_records, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    df = pd.DataFrame(log_records)
    df.to_csv(os.path.join(log_dir, "experiment_log.csv"), index=False)
    print(f"Saved results to {log_dir}")

def make_log_record(meta, obj_hist, grad_hist, dist_hist=None, weights=None, test_loss=None):
    return {
        **meta,
        "train_loss": obj_hist[-1],
        "test_loss": test_loss,
        "grad_norm": grad_hist[-1] if len(grad_hist) > 0 else None,
        "dist_to_opt": dist_hist[-1] if len(dist_hist) > 0 else None,
        "weights": weights.tolist() if hasattr(weights, "tolist") else list(weights),
        "obj_history": json.dumps([float(v) for v in obj_hist]),
        "grad_norm_history": json.dumps([float(v) for v in grad_hist]) if len(grad_hist) > 0 else None,
        "dist_to_opt_history": json.dumps([float(v) for v in dist_hist]) if len(dist_hist) > 0 else None
    }
