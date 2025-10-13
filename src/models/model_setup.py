import os; os.environ.setdefault("TOKENIZERS_PARALLELISM","false"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

def get_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    try:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)
    return model, tok
