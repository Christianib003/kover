# src/data/preprocess.py
import argparse, os, json
from src.utils.config import load_config

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(cfg_path):
    cfg = load_config(cfg_path)
    out_dir = cfg.paths["processed_dir"]
    rows = [
        {"id":"ex1","question":"What is a deductible?","answer":"A deductible is the amount you pay before coverage begins."},
        {"id":"ex2","question":"Is emergency room care covered?","answer":"Emergency services are typically covered; copays and deductibles may apply."},
        {"id":"ex3","question":"What is coinsurance?","answer":"Coinsurance is the percentage of costs you pay after meeting the deductible."}
    ]
    write_jsonl(os.path.join(out_dir,"train.jsonl"), rows[:2])
    write_jsonl(os.path.join(out_dir,"val.jsonl"),   rows[2:])
    write_jsonl(os.path.join(out_dir,"test.jsonl"),  rows[2:])
    print("Processed data written to", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
