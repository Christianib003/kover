import argparse, os, pandas as pd, numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def first_col(df: pd.DataFrame, names) -> pd.Series:
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series(dtype=str)

def load_insuranceqa() -> dict:
    ds = load_dataset("deccan-ai/insuranceQA-v2")
    def to_df(split):
        if split not in ds: return pd.DataFrame(columns=["question","answer"])
        raw = pd.DataFrame(ds[split])
        q = first_col(raw, ["input","question","q","query","question_text"])
        a = first_col(raw, ["output","answer","a","answer_text"])
        out = pd.DataFrame({"question": q, "answer": a}).dropna()
        out["question"] = out["question"].astype(str).str.strip().str.lower()
        out["answer"]   = out["answer"].astype(str).str.strip().str.lower()
        out = out[(out["question"]!="") & (out["answer"]!="")]
        out = out.drop_duplicates(subset=["question","answer"]).reset_index(drop=True)
        return out
    return {"train": to_df("train"), "val": to_df("validation"), "test": to_df("test")}

def main(raw_dir:str, processed_dir:str, val_ratio:float=0.1, seed:int=42):
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(processed_dir, exist_ok=True)
    splits = load_insuranceqa()

    base = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    base["id"] = np.arange(1, len(base)+1).astype(str)

    test = splits["test"].copy()
    test["id"] = np.arange(1, len(test)+1).astype(str)

    train_df, val_df = train_test_split(base, test_size=val_ratio, random_state=seed, shuffle=True)

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "validation.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    pd.concat([base.assign(split="base"), test.assign(split="test")]).to_csv(
        os.path.join(processed_dir, "insurance_qa_generative.csv"), index=False
    )
    print("rows:", {"train": len(train_df), "val": len(val_df), "test": len(test)})
    print("saved:", processed_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/insuranceqa")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.raw_dir, args.out_dir, args.val_ratio, args.seed)
