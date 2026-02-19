# Car Sales Price Modeling Project

|  |  |
|---|---|
| <img src="image.png" alt="University of San Andres logo" width="280"> | ## Final Project for the Machine Learning Course<br><br>**Institution:** University of San Andres, Argentina  \
**Program:** AI Engineer  \
**Academic Year:** 3rd year  \
**Course:** Machine Learning |

## Project Summary
This repository contains the final project developed for the Machine Learning course of the AI Engineer program at the University of San Andres (Argentina).

The project trains and evaluates regression models to estimate car sale prices using structured marketplace data and a configurable experimentation pipeline.

## Authors
- Antezana
- Giacometti

## Repository Structure
- `main.py`: CLI entrypoint to run experiments.
- `src/pipeline.py`: Orchestrates the full workflow.
- `configs/`: Data, feature, and model configuration modules.
- `models/`: Model implementations and data transformation module.
- `dataset/`: Input CSV dataset.
- `results/`: Saved model artifacts and per-run configuration snapshots.
- `logs/`: Execution logs.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run
Example run command:

```bash
python main.py \
  --model "configs/model/model_2.py" \
  --data "configs/data/data_params.py" \
  --features "configs/features/features_1.py"
```

## Output
Each run creates:
- A new folder in `results/` with the trained `model.pkl`.
- A `params_used_<n>.txt` file with run configuration references.
- A new log file in `logs/`.

## License
This project is distributed under the MIT License. See `LICENSE` for details.
