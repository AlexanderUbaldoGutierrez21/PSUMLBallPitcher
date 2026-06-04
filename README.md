# Major-League Baseball Pitcher Prediction

This project implements a polynomial regression model to predict pitcher performance metrics using MLB data. The model uses scikit-learn's LinearRegression with PolynomialFeatures (degree=2) to capture non-linear relationships between features x1 and x2 and the target variable y.

## Capabilities

- **Polynomial Regression**: Uses degree-2 polynomial features to model non-linear relationships
- **Automated Training & Prediction**: Loads training data, fits the model, and generates predictions on test data in a single script
- **CSV Output**: Saves predictions to a CSV file without headers or indices for easy submission

## Usage

Terminal MAC Run Script
```
python3 MLBALL_Pitcher.py
```

The script will:
1. Load training data from `data_train_midterm_problem6.csv`
2. Load test data from `data_test_midterm_problem6.csv`
3. Train a polynomial regression model (degree=2)
4. Generate predictions and save to `gutierrez_predictions_midterm_problem6.csv`

## Use Cases

### Educational Demonstration
Serves as a reference implementation for:
- Polynomial feature engineering with scikit-learn
- Linear regression on transformed features
- End-to-end ML pipeline in a single Python script

## Research Purposes

Designed for research purposes in an academic setting. The datasets included in this repository (`data_train_midterm_problem6.csv`, `data_test_midterm_problem6.csv`) are dummy datasets, do not represent real Major-League Baseball pitcher data. Penn State University (PSU), IST 557 Data Mining. Fall 2025.
