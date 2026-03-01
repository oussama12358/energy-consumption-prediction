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
├── energy_ml_project.py                    # Main project code
├── PJME_hourly.csv                        # Energy consumption dataset
├── README.md                               # Project documentation
├── LICENSE                                 # MIT License file
├── .gitignore                              # Git ignore file
├── 1_energy_exploratory_analysis.png       # Exploratory data analysis visualizations
├── 3_correlation_matrix.png                # Feature correlation heatmap
├── 4_model_comparison_v2.png              # Model performance comparison (RMSE, MAE, R²)
├── 5_predictions_vs_actual_v2.png         # Predicted vs actual values scatter plots
├── 6_cyclic_features.png                  # Cyclical feature visualizations
└── 7_feature_importance.png               # Feature importance from Random Forest
```

## Features

### Exploratory Data Analysis (EDA)
- Comprehensive data visualization and statistical analysis
- Data preprocessing and outlier detection (IQR method)
- Missing value handling and deduplication

### Feature Engineering
The following features are extracted from the datetime information:
- **Time-based Features**: Year, Month, Day, Hour, DayOfWeek, DayOfYear, WeekOfYear
- **Derived Features**:
  - `IsWeekend`: Binary flag (0=Weekday, 1=Weekend)
  - `IsPeakHour`: Binary flag for peak hours (6-9 AM and 5-10 PM)
  - `Season`: Categorical (0=Winter, 1=Spring, 2=Summer, 3=Autumn)

### Machine Learning Models
- **Linear Regression**: Baseline model for linear trend analysis
- **Decision Tree Regressor**: Non-linear model with controlled complexity
- **Random Forest Regressor**: Ensemble method for improved predictions

### Evaluation Metrics
- **RMSE (Root Mean Square Error)**: Measures average prediction error magnitude
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **R² Score**: Coefficient of determination (0-1), shows proportion of variance explained

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/energy-consumption-prediction.git
cd energy-consumption-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Download the energy consumption dataset from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

2. Place the CSV file in the project directory:
   - Supported filenames: `PJME_hourly.csv`, `pjm_hourly_est.csv`, or `AEP_hourly.csv`

3. Run the main script:
```bash
python energy_ml_project.py
```

4. The script will:
   - Load and explore the dataset
   - Perform data cleaning and feature engineering
   - Train three machine learning models
   - Generate 6 visualization files
   - Compare model performance

### Output

After running, you'll see:
- Console output with detailed progress and metrics
- 6 PNG visualization files in the project directory
- Performance comparison of all three models

## Results

The project generates:
- 6 visualization images showing EDA, correlation analysis, and model performance
- Performance metrics comparing different models
- Detailed predictions vs actual values analysis

### Generated Visualizations

> **Note:** The following image files were regenerated during the most recent script run and have been committed to Git:
> `4_model_comparison_v2.png`, `5_predictions_vs_actual_v2.png`, and `7_feature_importance.png`.
> These pertain to updated performance metrics and feature importance results.

### Generated Visualizations

#### 1. Energy Exploratory Analysis (`1_energy_exploratory_analysis.png`)
- **Energy Consumption Over Time**: Line plot showing energy consumption trends (first 1000 records)
- **Distribution of Energy Consumption**: Histogram showing the distribution pattern of energy usage
- **Average Consumption by Hour**: Hourly patterns revealing peak and off-peak hours
- **Average Consumption by Day of Week**: Daily patterns showing weekday vs weekend differences
- **Average Consumption by Month**: Monthly seasonal variations in energy consumption
- **Average Consumption by Season**: Seasonal comparison (Winter, Spring, Summer, Autumn)

#### 2. Correlation Matrix (`3_correlation_matrix.png`)
- Heatmap showing correlations between features and the target energy consumption variable
- Features include: Hour, DayOfWeek, Month, IsWeekend, IsPeakHour, Season
- Helps identify which features have the strongest relationship with energy consumption

#### 3. Model Comparison (`4_model_comparison_v2.png`)
Three comparison charts:
- **RMSE Comparison**: Root Mean Square Error for each model (lower is better)
- **MAE Comparison**: Mean Absolute Error for each model (lower is better)
- **R² Score Comparison**: Coefficient of determination for each model (higher is better)

Compares three models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

#### 4. Predictions vs Actual (`5_predictions_vs_actual_v2.png`)
- Scatter plots for each model comparing predicted vs actual energy consumption
- Red dashed line represents perfect predictions
- Shows how well each model generalizes to unseen data

#### 5. Cyclic Features (`6_cyclic_features.png`)
- Visualizations of cyclic patterns in time-based features
- Shows how energy consumption varies by hour, day of week, and season

#### 6. Feature Importance (`7_feature_importance.png`)
- Feature importance scores from the Random Forest model
- Identifies which features are most influential in predicting energy consumption

## Machine Learning Models

### 1. Linear Regression
- **Description**: Simple baseline model assuming linear relationships
- **Advantages**: Fast training, interpretable, good for baseline comparison
- **Use Case**: Identifying overall trends in energy consumption

### 2. Decision Tree Regressor
- **Description**: Non-linear model creating decision rules
- **Hyperparameters**: max_depth=15, min_samples_split=20, min_samples_leaf=10
- **Advantages**: Can capture non-linear patterns, easy to interpret
- **Use Case**: Finding complex patterns in energy usage

### 3. Random Forest Regressor
- **Description**: Ensemble method combining multiple decision trees
- **Hyperparameters**: n_estimators=100, max_depth=15, min_samples_split=20, min_samples_leaf=10
- **Advantages**: Reduces overfitting, handles non-linear relationships, provides feature importance
- **Use Case**: Production-grade predictions with good generalization

## Author

Oussama Sghir- 4th Year Engineering Student

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset source: [Kaggle - Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Course: AI Foundations 2025-2026
