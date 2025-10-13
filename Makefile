.PHONY: data prep app

PY=python

data:
	$(PY) -m src.prepare_data --raw_dir data/raw/insuranceqa --out_dir data/processed --val_ratio 0.1 --seed 42

prep:
	$(PY) -m src.data.preprocess --config configs/base.yaml

app:
	streamlit run app/main.py
