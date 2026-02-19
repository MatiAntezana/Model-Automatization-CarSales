# Car Sales Price Modeling Project

---

<table>
  <tr>
    <td width="36%" align="center" valign="middle">
      <img src="image.png" alt="University of San Andres logo" width="320" />
    </td>
    <td width="64%" valign="middle">
      <h2>Final Project: Machine Learning for Car Price Estimation</h2>
      <p><strong>Course:</strong> Machine Learning</p>
      <p><strong>Institution:</strong> University of San Andres, Argentina</p>
      <p><strong>Program:</strong> AI Engineer</p>
      <p><strong>Academic Year:</strong> 3rd year</p>
    </td>
  </tr>
</table>

## Project Summary
This repository contains the final project developed for the Machine Learning course of the AI Engineer program at the University of San Andres (Argentina).

The objective is to train and evaluate regression models that estimate used-car prices from structured marketplace data.

## Key Results (from report)
Source: [`Antezana_Giacommeti_report.pdf`](Antezana_Giacommeti_report.pdf)

- The report compares multiple predictive models on the same train/test protocol and evaluation metrics.
- **Random Forest** is identified as the best-performing model on the test set.
- According to the report discussion, Random Forest achieves stronger overall behavior in both **MAE** and **RMSE** comparisons.
- The predicted-vs-real price analysis shows better alignment for the selected model, while the report also highlights room for improvement on less frequent patterns and extreme cases.

## Authors
- Antezana
- Giacometti

## Repository Structure
- `main.py`: CLI entrypoint to run experiments.
- `src/pipeline.py`: Full workflow orchestration.
- `configs/`: Data, feature, and model configuration files.
- `models/`: Model definitions and transformation logic.
- `dataset/`: Input CSV dataset.
- `results/`: Saved model artifacts and run metadata.
- `logs/`: Execution logs.

## Requirements
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py \
  --model "configs/model/model_2.py" \
  --data "configs/data/data_params.py" \
  --features "configs/features/features_1.py"
```

## License
This project is distributed under the MIT License. See `LICENSE` for details.
