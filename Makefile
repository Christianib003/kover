.PHONY: prep train eval app
PY=python
prep: ; $(PY) -m src.data.preprocess --config configs/base.yaml
train: ; $(PY) -m src.modeling.train_tf --config configs/base.yaml
eval:  ; $(PY) -m src.eval.metrics --config configs/base.yaml
app:   ; $(PY) app.py
