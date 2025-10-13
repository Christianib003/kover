import argparse, os, json, pandas as pd
from src.utils.config import load_config

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_json(os.path.join(cfg.paths["processed_dir"], "test.jsonl"), lines=True)
    metrics = {"rows_test": len(df), "note": "wire real metrics after inference"}
    os.makedirs(cfg.paths["logs_dir"], exist_ok=True)
    with open(os.path.join(cfg.paths["logs_dir"], "metrics.json"), "w") as f: json.dump(metrics, f)
    print("Metrics written.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); main(args.config)
