# Makefile for the InsuranceQA Chatbot Project

# --- Variables ---
PYTHON := .env/bin/python # Or the name of your venv folder

# --- Targets ---

# Default target that runs when you just type 'make'
all: train-baseline

# Sets up the environment by installing dependencies
setup:
	@echo "Setting up the environment..."
	$(PYTHON) -m pip install -r requirements.txt

# Trains the baseline model with default hyperparameters
train-baseline:
	@echo "Training the baseline model..."
	TF_USE_LEGACY_KERAS=True $(PYTHON) src/train.py \
		--model_name "baseline_model" \
		--epochs 3 \
		--batch_size 8 \
		--learning_rate 2e-5

# Trains the model for the learning rate experiment
train-lr-experiment:
	@echo "Training the learning rate experiment model..."
	TF_USE_LEGACY_KERAS=True $(PYTHON) src/train.py \
		--model_name "lr_experiment_model" \
		--epochs 3 \
		--batch_size 8 \
		--learning_rate 5e-5

# Trains the final, optimized model
train-final:
	@echo "Training the final, optimized model..."
	TF_USE_LEGACY_KERAS=True $(PYTHON) src/train.py \
		--model_name "final_model" \
		--epochs 5 \
		--batch_size 16 \
		--learning_rate 5e-5

# Evaluates the champion model on the test set
evaluate:
	@echo "Evaluating the champion model on the test set..."
	TF_USE_LEGACY_KERAS=True $(PYTHON) src/run_evaluation.py --model_dir "models/lr_experiment_model"

# Runs the Streamlit application
app:
	@echo "Starting the Streamlit application..."
	@echo "Access the app at the URL provided in your terminal."
	$(PYTHON) -m streamlit run app/main.py

# Cleans up model artifacts (use with caution!)
clean:
	@echo "Cleaning up model directories..."
	rm -rf models/baseline_model
	rm -rf models/lr_experiment_model


.PHONY: all setup train-baseline train-lr-experiment app clean