import argparse, os, json
from src.utils.config import load_config
from src.utils.repro import set_seeds


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def main(cfg_path:str):
    cfg = load_config(cfg_path); set_seeds(cfg.get("seed",42))
    out = cfg.paths["processed_dir"]
    rows = [
        {"id":"ex1","question":"what is a deductible?","answer":"a deductible is the amount you pay before coverage begins."},
        {"id":"ex2","question":"is emergency room care covered?","answer":"emergency services are typically covered; copays/deductibles may apply."},
        {"id":"ex3","question":"what is coinsurance?","answer":"coinsurance is the percentage of costs you pay after the deductible."}
    ]
    write_jsonl(os.path.join(out,"train.jsonl"), rows[:2])
    write_jsonl(os.path.join(out,"validation.jsonl"), rows[2:])
    write_jsonl(os.path.join(out,"test.jsonl"), rows[2:])
    print("prep: wrote processed splits to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
