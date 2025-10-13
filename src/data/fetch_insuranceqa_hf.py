import os, json, argparse, hashlib
from datasets import load_dataset
from src.utils.config import load_config

RAW_FILENAMES_JSONL = {
    "answers": "answers.jsonl",
    "train":   "train.jsonl",
    "val":     "val.jsonl",
    "test":    "test.jsonl",
}

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    raw_dir = cfg.paths["raw_dir"]  # e.g., data/raw/insuranceqa
    os.makedirs(raw_dir, exist_ok=True)

    # Load the Hugging Face dataset (downloads to HF cache automatically).
    # Dataset card: deccan-ai/insuranceQA-v2
    ds = load_dataset("deccan-ai/insuranceQA-v2")  # splits: typically train/validation/test

    # Robust field detection: different mirrors sometimes use slightly different keys.
    # We normalize to:
    #  - question text  -> "question"
    #  - answer text    -> "answer"
    #  - (optional) id  -> "id"
    def norm_row(ex):
        q = ex.get("question") or ex.get("Question") or ex.get("q") or ""
        a = ex.get("answer")   or ex.get("Answer")   or ex.get("a") or ""
        _id = ex.get("id") or ex.get("_id")
        if not _id:
            # stable id from content
            _id = hashlib.md5((q + "||" + a).encode("utf-8")).hexdigest()[:12]
        return {"id": _id, "question": q.strip(), "answer": a.strip()}

    # Build per-split files and a deduplicated answer pool (answers.jsonl)
    answers_map = {}  # answer_text -> answer_id
    answers_rows = []

    def process_split(split_name_hf, out_name_jsonl):
        if split_name_hf not in ds:
            return
        rows = []
        for ex in ds[split_name_hf]:
            r = norm_row(ex)
            # assign a stable answer_id for the canonical pool
            atext = r["answer"]
            if atext not in answers_map:
                aid = hashlib.md5(atext.encode("utf-8")).hexdigest()[:12]
                answers_map[atext] = aid
                answers_rows.append({"answer_id": aid, "text": atext})
            r["gold_answer_id"] = answers_map[atext]
            rows.append({"id": r["id"], "question": r["question"], "gold_answer_id": r["gold_answer_id"]})
        write_jsonl(os.path.join(raw_dir, out_name_jsonl), rows)

    # HF often uses "validation"; our naming uses "val"
    process_split("train", RAW_FILENAMES_JSONL["train"])
    process_split("validation", RAW_FILENAMES_JSONL["val"])
    process_split("test", RAW_FILENAMES_JSONL["test"])

    # Write the answer pool last
    write_jsonl(os.path.join(raw_dir, RAW_FILENAMES_JSONL["answers"]), answers_rows)

    # Small summary
    print("Wrote raw files to:", raw_dir)
    for k, fn in RAW_FILENAMES_JSONL.items():
        p = os.path.join(raw_dir, fn)
        print(f"  - {fn}: {'OK' if os.path.exists(p) else 'MISSING'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
