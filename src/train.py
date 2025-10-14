import os, argparse, json, pandas as pd, tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from src.utils.config import load_config

def load_csv(proc_dir, split):
    return pd.read_csv(os.path.join(proc_dir, f"{split}_with_prompt.csv"))

def make_ds(df, tok, max_src, max_tgt, batch):
    X = tok(df["source"].tolist(),  truncation=True, padding=True, max_length=max_src, return_tensors="tf")
    y = tok(df["answer"].tolist(),  truncation=True, padding=True, max_length=max_tgt, return_tensors="tf")["input_ids"]
    return tf.data.Dataset.from_tensor_slices((dict(X), {"labels": y})).shuffle(2048).batch(batch).prefetch(tf.data.AUTOTUNE)

def main(cfg_path, override_model=None, epochs=None, lr=None, out_dir=None):
    cfg = load_config(cfg_path)
    model_name   = override_model or cfg.model["name"]
    max_src      = cfg.model["max_source_len"]
    max_tgt      = cfg.model["max_target_len"]
    batch_size   = cfg.train["batch_size"]
    num_epochs   = epochs or cfg.train["num_epochs"]
    learning_rate= lr or cfg.train["learning_rate"]

    proc_dir = cfg.paths["processed_dir"]
    save_dir = out_dir or os.path.join(cfg.paths["checkpoints_dir"], "baseline")
    os.makedirs(save_dir, exist_ok=True); os.makedirs(cfg.paths["logs_dir"], exist_ok=True)

    tok   = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_df = load_csv(proc_dir, "train")
    val_df   = load_csv(proc_dir, "validation")

    train_ds = make_ds(train_df, tok, max_src, max_tgt, batch_size)
    val_ds   = make_ds(val_df, tok, max_src, max_tgt, batch_size)

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt)

    ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, "best.keras"), save_best_only=True, monitor="val_loss")
    es   = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor="val_loss")

    hist = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, callbacks=[ckpt, es])

    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    with open(os.path.join(cfg.paths["logs_dir"], "train_history.json"), "w") as f:
        json.dump(hist.history, f)
    print("saved:", save_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    main(a.config, a.model, a.epochs, a.lr, a.out_dir)
