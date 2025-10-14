import os, argparse, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from transformers import AutoTokenizer


def add_prompt(df):
    df = df.copy()
    df["source"] = "question: " + df["question"].str.strip()
    return df


def tok_lengths(texts, tok, max_len=1024):
    return [len(tok(t, truncation=True, max_length=max_len)["input_ids"]) for t in texts]

def plot_hist(lengths, title, out):
    plt.figure(); plt.hist(lengths, bins=40); plt.title(title); plt.xlabel("tokens"); plt.ylabel("count")
    os.makedirs(os.path.dirname(out), exist_ok=True); plt.savefig(out); plt.close()

def main(proc_dir: str, model_name: str, plots_dir: str, logs_dir: str):
    train = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(proc_dir, "validation.csv"))
    test  = pd.read_csv(os.path.join(proc_dir, "test.csv"))

    train = add_prompt(train); val = add_prompt(val); test = add_prompt(test)

    train.to_csv(os.path.join(proc_dir, "train_with_prompt.csv"), index=False)
    val.to_csv(os.path.join(proc_dir, "validation_with_prompt.csv"), index=False)
    test.to_csv(os.path.join(proc_dir, "test_with_prompt.csv"), index=False)

    tok = AutoTokenizer.from_pretrained(model_name)
    q_lens = tok_lengths(train["question"].tolist(), tok)
    a_lens = tok_lengths(train["answer"].tolist(), tok)
    t_lens = tok_lengths(train["source"].tolist(), tok)

    stats = {
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "q_len": {"p50": int(np.percentile(q_lens,50)), "p90": int(np.percentile(q_lens,90)), "p95": int(np.percentile(q_lens,95)), "max": max(q_lens)},
        "a_len": {"p50": int(np.percentile(a_lens,50)), "p90": int(np.percentile(a_lens,90)), "p95": int(np.percentile(a_lens,95)), "max": max(a_lens)},
        "train_text_len": {"p50": int(np.percentile(t_lens,50)), "p90": int(np.percentile(t_lens,90)), "p95": int(np.percentile(t_lens,95)), "max": max(t_lens)},
        "suggested": {
            "max_source_len": int(np.percentile(q_lens,95)),
            "max_target_len": int(np.percentile(a_lens,95))
        }
    }

    plot_hist(q_lens, "Question token lengths", os.path.join(plots_dir, "q_token_lengths.png"))
    plot_hist(a_lens, "Answer token lengths",   os.path.join(plots_dir, "a_token_lengths.png"))
    plot_hist(t_lens, "Prompt token lengths",   os.path.join(plots_dir, "source_lengths.png"))

    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "clean_stats.json"), "w") as f: json.dump(stats, f, indent=2)
    print("suggested:", stats["suggested"])
    print("saved:", proc_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default="data/processed")
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--plots_dir", default="results/plots")
    ap.add_argument("--logs_dir", default="results/logs")
    args = ap.parse_args()
    main(args.proc_dir, args.model, args.plots_dir, args.logs_dir)
