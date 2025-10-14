import os, re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_USE_XLA", "0")

import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

def clean_text(s: str) -> str:
    s = s.replace("-lrb-", "(").replace("-rrb-", ")")
    s = s.replace("-lsb-", "[").replace("-rsb-", "]")
    s = re.sub(r"\s+", " ", s).strip()
    parts = re.split(r"(?<=[.!?])\s+", s)
    return " ".join(parts[:2]).strip()

class Chatbot:
    def __init__(self, model_dir: str, max_new_tokens: int = 160, max_source_len: int = 64):
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.max_new_tokens = max_new_tokens
        self.max_source_len = max_source_len

    def get_response(self, question_text: str) -> str:
        q = (question_text or "").strip()
        if not q:
            return "Please enter a question about insurance."
        src = f"question: {q.lower()}"
        enc = self.tok(src, return_tensors="tf", truncation=True, max_length=self.max_source_len)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=0.9,
            early_stopping=True,
        )
        ans = self.tok.decode(out[0], skip_special_tokens=True)
        return clean_text(ans)
