
PYTHON := .env/bin/python 


all: train-baseline

setup:
	@echo "Setting up the environment..."
	$(PYTHON) -m pip install -r requirements.txt

train-baseline:
	@echo "Training the baseline model..."
	TF_USE_LEGACY_KERAS=True $(PYTHON) src/train.py \
		--model_name "baseline_model" \
		--epochs 3 \
		--batch_size 8 \
		--learning_rate 2e-5

train-lr-experiment:
	@echo "Training the learning rate experiment model..."
	TF_USE_LEGACY_KERAS=True $(PYTHON) src/train.py \
		--model_name "lr_experiment_model" \
		--epochs 3 \
		--batch_size 8 \
		--learning_rate 5e-5

train-final:
	@echo "Training the final, optimized model..."
	TF_USE_LEGACY_KERAS=True $(PYTHON) src/train.py \
		--model_name "final_model" \
		--epochs 5 \
		--batch_size 16 \
		--learning_rate 5e-5

evaluate:
	@echo "Evaluating the champion model on the test set..."
	TF_USE_LEGACY_KERAS=True $(PYTHON) src/run_evaluation.py --model_dir "models/lr_experiment_model"

app:
	@echo "Starting the Streamlit application..."
	@echo "Access the app at the URL provided in your terminal."
	$(PYTHON) -m streamlit run app/main.py

clean:
	@echo "Cleaning up model directories..."
	rm -rf models/baseline_model
	rm -rf models/lr_experiment_model


.PHONY: all setup train-baseline train-lr-experiment app clean