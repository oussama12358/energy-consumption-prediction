# ================================================================
# ENERGY CONSUMPTION PREDICTION - ML PROJECT
# 4th Year Engineering - AI Foundations 2025-2026
# ================================================================
# DATASET: Hourly Energy Consumption
# SOURCE: Kaggle - https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
# ================================================================

# ================================================================
# 1. IMPORTING LIBRARIES
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("       ENERGY CONSUMPTION PREDICTION PROJECT")
print("="*70)
print("\nDataset: Hourly Energy Consumption (PJM Interconnection LLC)")
print("Source: Kaggle")
print("Link: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption")
print("="*70)

# ================================================================
# 2. LOADING THE DATASET
# ================================================================
print("\n[STEP 1] Loading the dataset...")
print("-" * 70)

# IMPORTANT: Download the dataset from Kaggle first
# Place the CSV file in the same folder as this code
# File name: PJME_hourly.csv (or adjust the filename below)

try:
    # Load the dataset
    # Try different possible filenames
    possible_files = ['pjm_hourly_est.csv', 'PJME_hourly.csv', 'AEP_hourly.csv']
    
    df = None
    for filename in possible_files:
        try:
            df = pd.read_csv(filename)
            print(f"✓ Dataset loaded successfully from: {filename}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("No dataset file found")
        
except FileNotFoundError:
    print("⚠ ERROR: Dataset file not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption")
    print("And place the CSV file in the same folder as this code.")
    print("Expected filenames: pjm_hourly_est.csv, PJME_hourly.csv, or AEP_hourly.csv")
    exit()

print(f"✓ Total records: {len(df):,}")
print(f"✓ Columns: {list(df.columns)}")

# ================================================================
# 3. DATASET PRESENTATION AND DESCRIPTION
# ================================================================
print("\n[STEP 2] Dataset Description and Characteristics")
print("-" * 70)

print("\n--- Dataset Overview ---")
print(f"Number of instances (rows): {df.shape[0]:,}")
print(f"Number of attributes (columns): {df.shape[1]}")
print(f"\nColumn names: {df.columns.tolist()}")

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Last 5 rows ---")
print(df.tail())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
missing_values = df.isnull().sum()
print(missing_values)
print(f"Total missing values: {missing_values.sum()}")

# ================================================================
# 4. DATA ANALYSIS AND ATTRIBUTE ANALYSIS
# ================================================================
print("\n[STEP 3] Data Analysis and Attribute Selection")
print("-" * 70)

# Convert Datetime column to datetime type
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Rename the energy column for easier access (adjust column name if different)
if 'PJME_MW' in df.columns:
    df.rename(columns={'PJME_MW': 'EnergyConsumption'}, inplace=True)
elif 'AEP_MW' in df.columns:
    df.rename(columns={'AEP_MW': 'EnergyConsumption'}, inplace=True)
else:
    # If column name is different, use the second column
    df.rename(columns={df.columns[1]: 'EnergyConsumption'}, inplace=True)

print("\n--- Target Variable ---")
print(f"Target: EnergyConsumption (Megawatts)")
print(f"Type: Numerical (Continuous)")
print(f"Purpose: Regression task")

print("\n--- Feature Engineering: Extracting Time-based Attributes ---")
# Extract time features from Datetime
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
df['DayOfYear'] = df['Datetime'].dt.dayofyear
df['WeekOfYear'] = df['Datetime'].dt.isocalendar().week

# Create additional derived features
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# Define peak hours (typically 6-9 AM and 5-10 PM)
df['IsPeakHour'] = ((df['Hour'] >= 6) & (df['Hour'] <= 9) | 
                     (df['Hour'] >= 17) & (df['Hour'] <= 22)).astype(int)

# Create season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn

df['Season'] = df['Month'].apply(get_season)

print("✓ Extracted features:")
print("  - Year, Month, Day, Hour")
print("  - DayOfWeek, DayOfYear, WeekOfYear")
print("  - IsWeekend (0=Weekday, 1=Weekend)")
print("  - IsPeakHour (0=Off-peak, 1=Peak)")
print("  - Season (0=Winter, 1=Spring, 2=Summer, 3=Autumn)")

print(f"\n--- Updated Dataset Shape ---")
print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")

# ================================================================
# 5. DATA CLEANING AND PREPARATION
# ================================================================
print("\n[STEP 4] Data Cleaning and Preparation")
print("-" * 70)

print(f"\nBefore cleaning:")
print(f"  - Total rows: {len(df):,}")
print(f"  - Missing values: {df.isnull().sum().sum()}")
print(f"  - Duplicate rows: {df.duplicated().sum()}")

# Handle missing values
if df['EnergyConsumption'].isnull().sum() > 0:
    print(f"\n✓ Handling missing values in EnergyConsumption...")
    # Option 1: Fill with mean
    df['EnergyConsumption'].fillna(df['EnergyConsumption'].mean(), inplace=True)
    # Option 2: Forward fill (uncomment if preferred)
    # df['EnergyConsumption'].fillna(method='ffill', inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove outliers using IQR method
Q1 = df['EnergyConsumption'].quantile(0.25)
Q3 = df['EnergyConsumption'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_count = ((df['EnergyConsumption'] < lower_bound) | 
                  (df['EnergyConsumption'] > upper_bound)).sum()
print(f"\n✓ Detected outliers: {outliers_count:,}")

# Remove outliers
df = df[(df['EnergyConsumption'] >= lower_bound) & 
        (df['EnergyConsumption'] <= upper_bound)]

print(f"\nAfter cleaning:")
print(f"  - Total rows: {len(df):,}")
print(f"  - Missing values: {df.isnull().sum().sum()}")
print(f"  - Duplicate rows: {df.duplicated().sum()}")
print(f"  - Outliers removed: {outliers_count:,}")

# Limit dataset size for faster training (optional - take last 10,000 records)
# Comment this out if you want to use the full dataset
if len(df) > 10000:
    df = df.tail(10000).reset_index(drop=True)
    print(f"\n✓ Dataset limited to last 10,000 records for efficient training")

print("\n✓ Dataset is now clean and ready for modeling!")

# ================================================================
# 6. DATA VISUALIZATION AND EXPLORATORY ANALYSIS
# ================================================================
print("\n[STEP 5] Data Visualization and Exploratory Analysis")
print("-" * 70)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Energy consumption over time (sample)
sample_data = df.head(1000)  # Plot first 1000 records
axes[0, 0].plot(sample_data['Datetime'], sample_data['EnergyConsumption'], 
                linewidth=1, alpha=0.7, color='#2E86AB')
axes[0, 0].set_title('Energy Consumption Over Time (Sample)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date', fontsize=11)
axes[0, 0].set_ylabel('Energy (MW)', fontsize=11)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribution of Energy Consumption
axes[0, 1].hist(df['EnergyConsumption'], bins=50, edgecolor='black', 
                alpha=0.7, color='#A23B72')
axes[0, 1].set_title('Distribution of Energy Consumption', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Energy Consumption (MW)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Average consumption by hour
hourly_avg = df.groupby('Hour')['EnergyConsumption'].mean()
axes[0, 2].plot(hourly_avg.index, hourly_avg.values, marker='o', 
                linewidth=2.5, markersize=7, color='#F18F01')
axes[0, 2].set_title('Average Energy Consumption by Hour', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Hour of Day', fontsize=11)
axes[0, 2].set_ylabel('Average Energy (MW)', fontsize=11)
axes[0, 2].set_xticks(range(0, 24, 2))
axes[0, 2].grid(True, alpha=0.3)

# 4. Average consumption by day of week
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_avg = df.groupby('DayOfWeek')['EnergyConsumption'].mean()
axes[1, 0].bar(range(7), daily_avg.values, color='#06A77D', 
               edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Average Energy Consumption by Day of Week', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Day of Week', fontsize=11)
axes[1, 0].set_ylabel('Average Energy (MW)', fontsize=11)
axes[1, 0].set_xticks(range(7))
axes[1, 0].set_xticklabels(day_names)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 5. Average consumption by month
monthly_avg = df.groupby('Month')['EnergyConsumption'].mean()
axes[1, 1].plot(monthly_avg.index, monthly_avg.values, marker='s', 
                linewidth=2.5, markersize=8, color='#C73E1D')
axes[1, 1].set_title('Average Energy Consumption by Month', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Month', fontsize=11)
axes[1, 1].set_ylabel('Average Energy (MW)', fontsize=11)
axes[1, 1].set_xticks(range(1, 13))
axes[1, 1].grid(True, alpha=0.3)

# 6. Average consumption by season
season_names = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
df['SeasonName'] = df['Season'].map(season_names)
season_avg = df.groupby('SeasonName')['EnergyConsumption'].mean()
axes[1, 2].bar(season_avg.index, season_avg.values, color='#6A4C93', 
               edgecolor='black', alpha=0.8)
axes[1, 2].set_title('Average Energy Consumption by Season', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Season', fontsize=11)
axes[1, 2].set_ylabel('Average Energy (MW)', fontsize=11)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('1_energy_exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Exploratory analysis saved as '1_energy_exploratory_analysis.png'")
plt.show()

# Boxplot for outlier visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].boxplot(df['EnergyConsumption'], vert=True)
axes[0].set_title('Box Plot of Energy Consumption', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Energy Consumption (MW)', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Weekday vs Weekend comparison
weekend_data = [df[df['IsWeekend']==0]['EnergyConsumption'], 
                df[df['IsWeekend']==1]['EnergyConsumption']]
axes[1].boxplot(weekend_data, labels=['Weekday', 'Weekend'])
axes[1].set_title('Energy Consumption: Weekday vs Weekend', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Energy Consumption (MW)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('2_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Box plots saved as '2_boxplots.png'")
plt.show()

# ================================================================
# 7. CORRELATION ANALYSIS
# ================================================================
print("\n[STEP 6] Correlation Analysis")
print("-" * 70)

# Select relevant features for correlation
correlation_cols = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 
                    'IsPeakHour', 'Season', 'EnergyConsumption']
corr_matrix = df[correlation_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 10, 'weight': 'bold'})
plt.title('Correlation Matrix - Energy Consumption Features', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('3_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Correlation matrix saved as '3_correlation_matrix.png'")
plt.show()

print("\n--- Most Correlated Features with Energy Consumption ---")
correlations = corr_matrix['EnergyConsumption'].sort_values(ascending=False)
print(correlations)

# ================================================================
# 8. PREPARING DATA FOR MACHINE LEARNING
# ================================================================
print("\n[STEP 7] Preparing Data for Machine Learning")
print("-" * 70)

# Select relevant features for modeling
# Remove non-numeric and redundant columns
features = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsPeakHour', 
            'Season', 'DayOfYear', 'WeekOfYear']

X = df[features]
y = df['EnergyConsumption']

print(f"\n✓ Features selected for modeling:")
for i, feat in enumerate(features, 1):
    print(f"   {i}. {feat}")

print(f"\n✓ Target variable: EnergyConsumption")
print(f"✓ Total samples: {len(X):,}")
print(f"✓ Number of features: {len(features)}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n--- Data Split ---")
print(f"Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Standardize features (important for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features standardized using StandardScaler")
print("✓ Data is ready for model training!")

# ================================================================
# 9. TRAINING THREE MACHINE LEARNING ALGORITHMS
# ================================================================
print("\n[STEP 8] Training Machine Learning Models")
print("="*70)

# Dictionary to store models and results
models = {}
results = {}

# ----------------------
# MODEL 1: Linear Regression
# ----------------------
print("\n1. LINEAR REGRESSION")
print("-" * 70)
print("Description: Simple linear approach, assumes linear relationships")
print("\n--- Algorithm Justification ---")
print("• Baseline model for comparison")
print("• Fast training and prediction")
print("• Good for identifying linear trends in energy consumption")
print("• No hyperparameters to tune (uses default optimal solution)")
print("\nTraining in progress...")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Calculate metrics
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

models['Linear Regression'] = lr_model
results['Linear Regression'] = {
    'RMSE': lr_rmse,
    'MAE': lr_mae,
    'R2': lr_r2,
    'predictions': lr_pred
}

print(f"✓ Model trained successfully!")
print(f"   RMSE (Root Mean Square Error): {lr_rmse:.2f} MW")
print(f"   MAE (Mean Absolute Error): {lr_mae:.2f} MW")
print(f"   R² Score: {lr_r2:.4f}")
print(f"   Interpretation: Model explains {lr_r2*100:.2f}% of variance")

# ----------------------
# MODEL 2: Decision Tree Regressor
# ----------------------
print("\n2. DECISION TREE REGRESSOR")
print("-" * 70)
print("Description: Non-linear model, creates decision rules")
print("\n--- Hyperparameter Selection and Justification ---")
print("• max_depth=15: Limits tree depth to prevent overfitting")
print("• min_samples_split=20: Requires 20 samples to split a node")
print("• min_samples_leaf=10: Minimum 10 samples per leaf for stability")
print("• random_state=42: For reproducibility")
print("\nRationale: These parameters balance model complexity and generalization")
print("Training in progress...")

dt_model = DecisionTreeRegressor(max_depth=15, min_samples_split=20, 
                                  min_samples_leaf=10, random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

# Calculate metrics
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

models['Decision Tree'] = dt_model
results['Decision Tree'] = {
    'RMSE': dt_rmse,
    'MAE': dt_mae,
    'R2': dt_r2,
    'predictions': dt_pred
}

print(f"✓ Model trained successfully!")
print(f"   RMSE (Root Mean Square Error): {dt_rmse:.2f} MW")
print(f"   MAE (Mean Absolute Error): {dt_mae:.2f} MW")
print(f"   R² Score: {dt_r2:.4f}")
print(f"   Interpretation: Model explains {dt_r2*100:.2f}% of variance")

# ----------------------
# MODEL 3: Random Forest Regressor
# ----------------------
print("\n3. RANDOM FOREST REGRESSOR")
print("-" * 70)
print("Description: Ensemble method, combines multiple decision trees")
print("\n--- Hyperparameter Selection and Justification ---")
print("• n_estimators=100: Number of trees in the forest")
print("• max_depth=15: Maximum depth of each tree")
print("• min_samples_split=20: Minimum samples to split a node")
print("• min_samples_leaf=10: Minimum samples per leaf")
print("• random_state=42: For reproducibility")
print("• n_jobs=-1: Use all CPU cores for parallel processing")
print("\nRationale: Multiple trees reduce overfitting and improve accuracy")
print("Training in progress...")

rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, 
                                  min_samples_split=20, min_samples_leaf=10,
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# Calculate metrics
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

models['Random Forest'] = rf_model
results['Random Forest'] = {
    'RMSE': rf_rmse,
    'MAE': rf_mae,
    'R2': rf_r2,
    'predictions': rf_pred
}

print(f"✓ Model trained successfully!")
print(f"   RMSE (Root Mean Square Error): {rf_rmse:.2f} MW")
print(f"   MAE (Mean Absolute Error): {rf_mae:.2f} MW")
print(f"   R² Score: {rf_r2:.4f}")
print(f"   Interpretation: Model explains {rf_r2*100:.2f}% of variance")

# ================================================================
# 10. COMPARISON OF RESULTS
# ================================================================
print("\n[STEP 9] Comparing Model Performance")
print("="*70)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE (MW)': [results[m]['RMSE'] for m in results.keys()],
    'MAE (MW)': [results[m]['MAE'] for m in results.keys()],
    'R² Score': [results[m]['R2'] for m in results.keys()]
})

print("\n--- MODEL COMPARISON TABLE ---")
print("="*70)
print(comparison_df.to_string(index=False))
print("="*70)

# Find best model based on RMSE (lower is better)
best_model_idx = comparison_df['RMSE (MW)'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_rmse = comparison_df.loc[best_model_idx, 'RMSE (MW)']
best_r2 = comparison_df.loc[best_model_idx, 'R² Score']

print(f"\n BEST PERFORMING MODEL: {best_model_name}")
print(f"   ✓ Lowest RMSE: {best_rmse:.2f} MW")
print(f"   ✓ R² Score: {best_r2:.4f}")
print(f"   ✓ This model explains {best_r2*100:.2f}% of the variance in energy consumption")

# Model ranking
print(f"\n--- MODEL RANKING (by RMSE) ---")
ranked = comparison_df.sort_values('RMSE (MW)')
for i, (idx, row) in enumerate(ranked.iterrows(), 1):
    print(f"   {i}. {row['Model']} - RMSE: {row['RMSE (MW)']:.2f} MW")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# RMSE comparison
axes[0].bar(comparison_df['Model'], comparison_df['RMSE (MW)'], 
            color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_title('RMSE Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('RMSE (MW)', fontsize=12)
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparison_df['RMSE (MW)']):
    axes[0].text(i, v + 20, f'{v:.1f}', ha='center', fontweight='bold')

# MAE comparison
axes[1].bar(comparison_df['Model'], comparison_df['MAE (MW)'], 
            color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_title('MAE Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('MAE (MW)', fontsize=12)
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparison_df['MAE (MW)']):
    axes[1].text(i, v + 15, f'{v:.1f}', ha='center', fontweight='bold')

# R² Score comparison
axes[2].bar(comparison_df['Model'], comparison_df['R² Score'], 
            color=colors, edgecolor='black', linewidth=1.5)
axes[2].set_title('R² Score Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('R² Score', fontsize=12)
axes[2].tick_params(axis='x', rotation=15)
axes[2].set_ylim([0, 1])
axes[2].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparison_df['R² Score']):
    axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('4_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison saved as '4_model_comparison.png'")
plt.show()

# Predictions vs Actual values
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, model_results) in enumerate(results.items()):
    # Scatter plot
    axes[idx].scatter(y_test, model_results['predictions'], alpha=0.4, s=20)
    
    # Perfect prediction line
    min_val = min(y_test.min(), model_results['predictions'].min())
    max_val = max(y_test.max(), model_results['predictions'].max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    axes[idx].set_title(f'{model_name}\nR² = {model_results["R2"]:.4f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Actual Energy Consumption (MW)', fontsize=11)
    axes[idx].set_ylabel('Predicted Energy Consumption (MW)', fontsize=11)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('5_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("✓ Predictions vs Actual saved as '5_predictions_vs_actual.png'")