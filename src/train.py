import os, json, time, csv, argparse, pandas as pd, tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from src.utils.config import load_config

def load_split(proc_dir: str, split: str) -> pd.DataFrame:
    p = os.path.join(proc_dir, f"{split}_with_prompt.csv")
    if not os.path.exists(p): 
        p = os.path.join(proc_dir, f"{split}.csv")
    df = pd.read_csv(p)
    if "source" not in df.columns:  
        df["source"] = "question: " + df["question"].astype(str).str.strip()
    return df[["source", "answer"]].dropna()

def make_ds(df, tok, max_src, max_tgt, batch):
    X = tok(df["source"].tolist(), truncation=True, padding=True, max_length=max_src, return_tensors="tf")
    y_ids = tok(df["answer"].tolist(), truncation=True, padding=True, max_length=max_tgt, return_tensors="tf")["input_ids"]
    pad_id = tok.pad_token_id
    labels = tf.where(tf.equal(y_ids, pad_id), -100, y_ids)
    ds = tf.data.Dataset.from_tensor_slices((dict(X), {"labels": labels}))
    return ds.shuffle(2048).batch(batch).prefetch(tf.data.AUTOTUNE)


def main(cfg_path, override_model=None, epochs=None, lr=None, out_dir=None, run_name=None):
    cfg = load_config(cfg_path)
    model_name    = override_model or cfg.model["name"]
    max_src       = int(cfg.model["max_source_len"])
    max_tgt       = int(cfg.model["max_target_len"])
    batch_size    = int(cfg.train["batch_size"])
    num_epochs    = int(epochs or cfg.train["num_epochs"])
    learning_rate = float(lr or cfg.train["learning_rate"])

    proc_dir   = cfg.paths["processed_dir"]
    ckpt_root  = cfg.paths["checkpoints_dir"]
    logs_root  = cfg.paths["logs_dir"]
    os.makedirs(ckpt_root, exist_ok=True); os.makedirs(logs_root, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    try:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)

    train_df = load_split(proc_dir, "train")
    val_df   = load_split(proc_dir, "validation")

    train_ds = make_ds(train_df, tok, max_src, max_tgt, batch_size)
    val_ds   = make_ds(val_df,   tok, max_src, max_tgt, batch_size)

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt)

    run_id  = run_name or f"{model_name.replace('/','_')}_{int(time.time())}"
    save_dir = out_dir or os.path.join(ckpt_root, run_id)
    os.makedirs(save_dir, exist_ok=True)

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(save_dir, "best.keras"), save_best_only=True, monitor="val_loss"
    )
    es_cb = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss")

    hist = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[ckpt_cb, es_cb])

    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    meta = {
        "run_id": run_id,
        "model": model_name,
        "max_src": max_src,
        "max_tgt": max_tgt,
        "batch": batch_size,
        "epochs": num_epochs,
        "lr": learning_rate,
        "val_loss": float(hist.history["val_loss"][-1]),
    }
    with open(os.path.join(save_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    runs_csv = os.path.join(logs_root, "runs.csv")
    write_header = not os.path.exists(runs_csv)
    with open(runs_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta.keys()))
        if write_header: w.writeheader()
        w.writerow(meta)

    with open(os.path.join(logs_root, "train_history.json"), "w") as f:
        json.dump(hist.history, f)

    print("saved:", save_dir, "| run:", run_id)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--run_name", default=None)
    a = ap.parse_args()
    main(a.config, a.model, a.epochs, a.lr, a.out_dir, a.run_name)
