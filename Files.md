# Files and directories

### Directory structure:
- `results` – stores model output and trained models. Due to large file sizes, models are not saved at this stage.
- `data` – contains datasets. In practice, only one dataset is used — the one that yields the best results.
- `scripts_<model_name>` – contains Python scripts that run the respective models.
  - `first_model_...py` – model with a single set of hyperparameters
  - `tuning_models_...py` – scripts for hyperparameter tuning
  - `best_model_...py` – model providing the best performance
  - `log_...txt` – training logs

### Required Python dependencies:

*Will be in a while* 

