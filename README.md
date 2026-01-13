# Energy Consumption Prediction Using Machine Learning

A machine learning project that predicts hourly energy consumption using various regression models.

## Project Overview

This project uses historical energy consumption data to build and compare multiple machine learning models for predicting future energy usage patterns. The project is part of a 4th-year engineering AI Foundations course (2025-2026).

## Dataset

- **Source**: Kaggle - [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- **Data**: Hourly energy consumption from PJM Interconnection LLC (PJME)
- **File**: `PJME_hourly.csv`

## Project Structure

```
├── energy_ml_project.py           # Main project code
├── PJME_hourly.csv               # Energy consumption dataset
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
└── images/                        # Generated visualizations
    ├── 1_energy_exploratory_analysis.png
    ├── 2_boxplots.png
    ├── 3_correlation_matrix.png
    ├── 4_model_comparison.png
    └── 5_predictions_vs_actual.png
```

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and visualization
- Multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Model comparison and evaluation
- Performance metrics (MSE, MAE, R² Score)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/energy-consumption-prediction.git
cd energy-consumption-prediction
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
2. Place `PJME_hourly.csv` in the project directory
3. Run the project:
```bash
python energy_ml_project.py
```

## Results

The project generates:
- 5 visualization images showing EDA and model performance
- Performance metrics comparing different models
- Predictions vs actual values plots

## Author

Your Name - 4th Year Engineering Student

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset source: [Kaggle - Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Course: AI Foundations 2025-2026
