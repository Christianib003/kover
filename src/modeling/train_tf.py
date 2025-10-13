import argparse, os, json, pandas as pd, tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from src.utils.config import load_config
from src.utils.repro import set_seeds

def load_jsonl(p): return pd.read_json(p, lines=True)

def make_ds(df, tok, max_src, max_tgt, batch):
    X = tok(df["question"].tolist(), truncation=True, padding=True, max_length=max_src, return_tensors="tf")
    y = tok(df["answer"].tolist(),   truncation=True, padding=True, max_length=max_tgt, return_tensors="tf")["input_ids"]
    ds = tf.data.Dataset.from_tensor_slices((dict(X), {"labels": y})).batch(batch)
    return ds

def main(cfg_path):
    cfg = load_config(cfg_path); set_seeds(cfg.d.get("seed", 42))
    tok = AutoTokenizer.from_pretrained(cfg.model["name"])
    model = TFAutoModelForSeq2SeqLM.from_pretrained(cfg.model["name"])
    train = load_jsonl(os.path.join(cfg.paths["processed_dir"], "train.jsonl"))
    val   = load_jsonl(os.path.join(cfg.paths["processed_dir"], "val.jsonl"))
    train_ds = make_ds(train, tok, cfg.model["max_source_len"], cfg.model["max_target_len"], cfg.train["batch_size"])
    val_ds   = make_ds(val, tok,   cfg.model["max_source_len"], cfg.model["max_target_len"], cfg.train["batch_size"])
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.train["learning_rate"])
    model.compile(optimizer=opt)
    os.makedirs(cfg.paths["checkpoints_dir"], exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(cfg.paths["checkpoints_dir"], "baseline.keras"), save_best_only=True, monitor="val_loss")
    es   = tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True, monitor="val_loss")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg.train["num_epochs"], callbacks=[ckpt, es])
    os.makedirs(cfg.paths["logs_dir"], exist_ok=True)
    with open(os.path.join(cfg.paths["logs_dir"], "train_history.json"), "w") as f: json.dump(hist.history, f)
    print("Training complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); main(args.config)
