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

## Introduction
This repository contains the final project developed for the Machine Learning course in the AI Engineer program at the University of San Andres (Argentina).

The project goal is to estimate used-car prices using supervised regression models and a reproducible experimentation pipeline.

## Technologies & Skills
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Random Forest, K-Nearest Neighbors, Linear Regression, Extra Trees)
- **Model Serialization**: Joblib
- **Core Concepts**: Regression Modeling, Hyperparameter Tuning, Feature Engineering

## Dataset Used
- File: `dataset/pf_suvs_i302_1s2024.csv`
- Size: 22,377 rows and 16 columns
- Target variable: `Price` (original column: `Precio`)
- Main attributes: brand, model, year, version, fuel type, transmission, mileage, currency, seller type, and related listing metadata.

## Data Treatment
Based on the methodology described in [`Antezana_Giacommeti_report.pdf`](Antezana_Giacommeti_report.pdf), the preprocessing stage focused on:

- Cleaning irregular values and correcting inconsistent records.
- Reviewing missing data and preserving useful variables for prediction.
- Detecting and controlling outliers using quartile-based criteria (Q1/Q3), especially for price and mileage.
- Harmonizing currency values for consistent price modeling.
- Splitting data into train/test sets for fair model comparison.

## Models Evaluated
The project evaluated four regression models to estimate car prices:
- K-Nearest Neighbors (KNN)
- Linear Regression
- Extra Trees Regressor
- Random Forest Regressor

The **Random Forest Regressor** yielded the best performance metrics, achieving the lowest Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The optimal hyperparameters found using GridSearchCV were:
- Minimum samples to split a node: 15
- Number of trees: 200
- Out-of-bag (OOB) score: True

## Key Findings
A detailed analysis was conducted on the impact of mileage and year of manufacture on the estimated price:
- **Mileage**: The models successfully captured the trend that the price of a vehicle decreases as its accumulated mileage increases.
- **Year of Manufacture**: Generally, newer cars have higher prices. However, the model had more difficulty isolating this effect independently, suggesting potential correlation with other features or the need for a larger sample of certain manufacturing years.

## Future Work
Based on the results, proposed improvements and future directions include:
- **Dataset Expansion**: Adding more complete records and reducing missing data to enhance the models' robustness across a wider range of vehicle types, ages, and mileage.
- **Further Hyperparameter Tuning**: Continuing the search for better parameters or exploring new architectures (e.g., more complex ensembles or neural networks).
- **Feature Importance**: Further analysis of the impact of other features such as fuel type, seller type, and engine characteristics on the final vehicle price.

## Creator & Authors
- **Creator**: Matias Luciano Antezana
- **Co-Author**: Mateo Giacometti

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
