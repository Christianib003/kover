.PHONY: data app format
PY=python

data:
	$(PY) -m src.prepare_data --raw_dir data/raw/insuranceqa --out_dir data/processed --val_ratio 0.1 --seed 42

format:
	$(PY) -m src.data.format_and_eda --proc_dir data/processed --model google/flan-t5-small --plots_dir results/plots --logs_dir results/logs

app:
	streamlit run app/main.py
