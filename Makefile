.PHONY: prep app

PY=python

prep:
	$(PY) -m src.data.preprocess --config configs/base.yaml

app:
	streamlit run app/main.py
