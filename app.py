import gradio as gr
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

MODEL = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(MODEL)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

def respond(q):
    enc = tok(q, return_tensors="tf", truncation=True, max_length=256)
    out = model.generate(**enc, max_new_tokens=64)
    return tok.decode(out[0], skip_special_tokens=True)

demo = gr.Interface(fn=respond, inputs=gr.Textbox(label="Ask about insurance"),
                    outputs="text", title="InsuranceQA Chatbot (Prototype)")
if __name__ == "__main__":
    demo.launch()
