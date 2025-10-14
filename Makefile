.PHONY: data app format model train train-flan
PY=python

data:
	$(PY) -m src.prepare_data --raw_dir data/raw/insuranceqa --out_dir data/processed --val_ratio 0.1 --seed 42

format:
	$(PY) -m src.data.format_and_eda --proc_dir data/processed --model google/flan-t5-small --plots_dir results/plots --logs_dir results/logs

train:
	$(PY) -m src.train --config configs/base.yaml --model t5-small

train-flan:
	$(PY) -m src.train --config configs/base.yaml --model google/flan-t5-small --run_name flan_t5small_baseline

model:
	TOKENIZERS_PARALLELISM=false OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES $(PY) -c "from src.models.model_setup import get_model_and_tokenizer as f; m,t=f('t5-small'); print('ok')"

app:
	streamlit run app/main.py
