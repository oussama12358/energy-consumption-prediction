# ================================================================
# ENERGY CONSUMPTION PREDICTION - ML PROJECT v2.0
# 4th Year Engineering - AI Foundations 2025-2026
# ================================================================
# DATASET: Hourly Energy Consumption
# SOURCE: Kaggle - https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
#
# IMPROVEMENTS (v2.0) - Based on expert feedback:
#   1. Cyclic features (sin/cos) for hour, month, day of week
#   2. XGBoost model added
#   3. LightGBM model added
#   4. Hyperparameter optimization with Optuna
#   5. Train/test gap analysis to detect overfitting
#   6. Temporal split (no shuffle) for time series integrity
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

# New imports for v2.0
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not installed. Run: pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠ Optuna not installed. Run: pip install optuna")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("       ENERGY CONSUMPTION PREDICTION PROJECT  v2.0")
print("="*70)
print("\nDataset: Hourly Energy Consumption (PJM Interconnection LLC)")
print("Source: Kaggle")
print("="*70)

# ================================================================
# 2. LOADING THE DATASET
# ================================================================
print("\n[STEP 1] Loading the dataset...")
print("-" * 70)

try:
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
    exit()

print(f"✓ Total records: {len(df):,}")
print(f"✓ Columns: {list(df.columns)}")

# ================================================================
# 3. DATASET DESCRIPTION
# ================================================================
print("\n[STEP 2] Dataset Description")
print("-" * 70)
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print(df.describe())

# ================================================================
# 4. FEATURE ENGINEERING  (v2.0 - with cyclic features)
# ================================================================
print("\n[STEP 3] Feature Engineering")
print("-" * 70)

df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)  # important for time series

# Rename target column
if 'PJME_MW' in df.columns:
    df.rename(columns={'PJME_MW': 'EnergyConsumption'}, inplace=True)
elif 'AEP_MW' in df.columns:
    df.rename(columns={'AEP_MW': 'EnergyConsumption'}, inplace=True)
else:
    df.rename(columns={df.columns[1]: 'EnergyConsumption'}, inplace=True)

# Basic time features
df['Year']      = df['Datetime'].dt.year
df['Month']     = df['Datetime'].dt.month
df['Day']       = df['Datetime'].dt.day
df['Hour']      = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['DayOfYear'] = df['Datetime'].dt.dayofyear
df['WeekOfYear']= df['Datetime'].dt.isocalendar().week

# Binary features
df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
df['IsPeakHour'] = ((df['Hour'] >= 6) & (df['Hour'] <= 9) |
                    (df['Hour'] >= 17) & (df['Hour'] <= 22)).astype(int)

# Season
def get_season(month):
    if month in [12, 1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8]: return 2
    else: return 3
df['Season'] = df['Month'].apply(get_season)

# ============================================================
# NEW v2.0 - CYCLIC FEATURES (sin/cos encoding)
# ============================================================
# Why? Hour=23 and Hour=0 are close in reality but far apart as numbers.
# Cyclic encoding captures this circular/periodic nature of time.
print("\n✓ Adding cyclic (sin/cos) features for time periodicity...")

df['Hour_sin']      = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos']      = np.cos(2 * np.pi * df['Hour'] / 24)

df['Month_sin']     = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos']     = np.cos(2 * np.pi * df['Month'] / 12)

df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

print("  - Hour_sin, Hour_cos")
print("  - Month_sin, Month_cos")
print("  - DayOfWeek_sin, DayOfWeek_cos")
print("  - DayOfYear_sin, DayOfYear_cos")

print(f"\n✓ Total features after engineering: {df.shape[1]}")

# ================================================================
# 5. DATA CLEANING
# ================================================================
print("\n[STEP 4] Data Cleaning")
print("-" * 70)

print(f"Before cleaning: {len(df):,} rows | "
      f"Missing: {df.isnull().sum().sum()} | "
      f"Duplicates: {df.duplicated().sum()}")

# Handle missing values
if df['EnergyConsumption'].isnull().sum() > 0:
    # Forward fill is better for time series than mean fill
    df['EnergyConsumption'].fillna(method='ffill', inplace=True)
    df['EnergyConsumption'].fillna(method='bfill', inplace=True)

df.drop_duplicates(inplace=True)

# Outlier removal using IQR
Q1, Q3 = df['EnergyConsumption'].quantile(0.25), df['EnergyConsumption'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['EnergyConsumption'] >= Q1 - 1.5*IQR) &
        (df['EnergyConsumption'] <= Q3 + 1.5*IQR)]

print(f"After cleaning:  {len(df):,} rows | "
      f"Missing: {df.isnull().sum().sum()}")

# Use last 50,000 records (v2.0: more data = better generalization)
if len(df) > 50000:
    df = df.tail(50000).reset_index(drop=True)
    print(f"✓ Using last 50,000 records for training")

# ================================================================
# 6. VISUALIZATIONS
# ================================================================
print("\n[STEP 5] Exploratory Data Visualization")
print("-" * 70)

df['SeasonName'] = df['Season'].map({0:'Winter',1:'Spring',2:'Summer',3:'Autumn'})

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Energy Consumption - Exploratory Analysis', fontsize=16, fontweight='bold')

sample_data = df.tail(1000)
axes[0,0].plot(sample_data['Datetime'], sample_data['EnergyConsumption'],
               linewidth=1, alpha=0.7, color='#2E86AB')
axes[0,0].set_title('Energy Over Time (Last 1000 records)')
axes[0,0].set_xlabel('Date'); axes[0,0].tick_params(axis='x', rotation=45)

axes[0,1].hist(df['EnergyConsumption'], bins=50, edgecolor='black', alpha=0.7, color='#A23B72')
axes[0,1].set_title('Distribution of Energy Consumption')
axes[0,1].set_xlabel('Energy (MW)')

hourly_avg = df.groupby('Hour')['EnergyConsumption'].mean()
axes[0,2].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2.5, color='#F18F01')
axes[0,2].set_title('Avg Consumption by Hour')
axes[0,2].set_xlabel('Hour of Day'); axes[0,2].set_xticks(range(0,24,2))

day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
daily_avg = df.groupby('DayOfWeek')['EnergyConsumption'].mean()
axes[1,0].bar(range(7), daily_avg.values, color='#06A77D', edgecolor='black', alpha=0.8)
axes[1,0].set_title('Avg Consumption by Day of Week')
axes[1,0].set_xticks(range(7)); axes[1,0].set_xticklabels(day_names)

monthly_avg = df.groupby('Month')['EnergyConsumption'].mean()
axes[1,1].plot(monthly_avg.index, monthly_avg.values, marker='s', linewidth=2.5, color='#C73E1D')
axes[1,1].set_title('Avg Consumption by Month')
axes[1,1].set_xlabel('Month'); axes[1,1].set_xticks(range(1,13))

season_avg = df.groupby('SeasonName')['EnergyConsumption'].mean()
axes[1,2].bar(season_avg.index, season_avg.values, color='#6A4C93', edgecolor='black', alpha=0.8)
axes[1,2].set_title('Avg Consumption by Season')

plt.tight_layout()
plt.savefig('1_energy_exploratory_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 1_energy_exploratory_analysis.png")
plt.show()

# NEW v2.0 - Cyclic feature visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Cyclic Features - Hour Encoding', fontsize=14, fontweight='bold')
axes[0].scatter(df['Hour'], df['EnergyConsumption'], alpha=0.05, s=5, color='steelblue')
axes[0].set_title('Raw Hour vs Energy (linear - loses circularity)')
axes[0].set_xlabel('Hour'); axes[0].set_ylabel('Energy (MW)')

axes[1].scatter(df['Hour_sin'], df['Hour_cos'], c=df['EnergyConsumption'],
                cmap='plasma', alpha=0.3, s=5)
axes[1].set_title('Hour_sin vs Hour_cos (cyclic - preserves circularity)')
axes[1].set_xlabel('Hour_sin'); axes[1].set_ylabel('Hour_cos')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Energy (MW)')
plt.tight_layout()
plt.savefig('6_cyclic_features.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 6_cyclic_features.png")
plt.show()

# ================================================================
# 7. CORRELATION ANALYSIS
# ================================================================
print("\n[STEP 6] Correlation Analysis")
print("-" * 70)

correlation_cols = ['Hour', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos',
                    'DayOfWeek', 'IsWeekend', 'IsPeakHour', 'Season', 'EnergyConsumption']
corr_matrix = df[correlation_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1.5, annot_kws={'size': 9})
plt.title('Correlation Matrix (v2.0 - with cyclic features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('3_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 3_correlation_matrix.png")
plt.show()

# ================================================================
# 8. PREPARING DATA FOR ML
# ================================================================
print("\n[STEP 7] Preparing Data for Machine Learning")
print("-" * 70)

# v2.0: Use cyclic features + original features
features = [
    # Original features
    'DayOfYear', 'WeekOfYear', 'IsWeekend', 'IsPeakHour', 'Season', 'Year',
    # NEW cyclic features
    'Hour_sin', 'Hour_cos',
    'Month_sin', 'Month_cos',
    'DayOfWeek_sin', 'DayOfWeek_cos',
    'DayOfYear_sin', 'DayOfYear_cos'
]

X = df[features]
y = df['EnergyConsumption']

print(f"✓ Features: {features}")
print(f"✓ Total samples: {len(X):,} | Features: {len(features)}")

# ============================================================
# NEW v2.0 - TEMPORAL SPLIT (no shuffle for time series)
# ============================================================
# Why no shuffle? Shuffling time series data causes data leakage.
# The model would "see the future" during training.
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n✓ Temporal split (no shuffle - time series integrity):")
print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.0f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models  = {}
results = {}

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, use_scaled=True):
    """Train, evaluate, and store results. Also prints train vs test gap."""
    Xtr = X_tr if use_scaled else X_train
    Xte = X_te if use_scaled else X_test
    model.fit(Xtr, y_tr)
    train_pred = model.predict(Xtr)
    test_pred  = model.predict(Xte)

    train_r2 = r2_score(y_tr, train_pred)
    test_r2  = r2_score(y_te, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, test_pred))
    test_mae  = mean_absolute_error(y_te, test_pred)

    # NEW v2.0 - Train/Test gap analysis
    gap = train_r2 - test_r2
    overfitting_flag = "⚠ Overfitting!" if gap > 0.1 else "✓ Good generalization"

    print(f"\n  Train R²: {train_r2:.4f}  |  Test R²: {test_r2:.4f}  |  Gap: {gap:.4f}  →  {overfitting_flag}")
    print(f"  RMSE: {test_rmse:.2f} MW  |  MAE: {test_mae:.2f} MW")

    models[name]  = model
    results[name] = {
        'RMSE': test_rmse, 'MAE': test_mae,
        'R2': test_r2, 'Train_R2': train_r2,
        'Gap': gap, 'predictions': test_pred
    }

# ================================================================
# 9. MODEL TRAINING
# ================================================================
print("\n[STEP 8] Training Models")
print("="*70)

# --- MODEL 1: Linear Regression (baseline) ---
print("\n1. LINEAR REGRESSION (Baseline)")
print("-" * 50)
print("   Purpose: Establish a baseline. Expected low performance since")
print("   energy consumption has non-linear relationships.")
evaluate_model('Linear Regression', LinearRegression(),
               X_train_scaled, X_test_scaled, y_train, y_test)

# --- MODEL 2: Decision Tree ---
print("\n2. DECISION TREE REGRESSOR")
print("-" * 50)
evaluate_model('Decision Tree',
               DecisionTreeRegressor(max_depth=15, min_samples_split=20,
                                     min_samples_leaf=10, random_state=42),
               X_train_scaled, X_test_scaled, y_train, y_test)

# --- MODEL 3: Random Forest ---
print("\n3. RANDOM FOREST REGRESSOR")
print("-" * 50)
evaluate_model('Random Forest',
               RandomForestRegressor(n_estimators=100, max_depth=15,
                                     min_samples_split=20, min_samples_leaf=10,
                                     random_state=42, n_jobs=-1),
               X_train_scaled, X_test_scaled, y_train, y_test)

# ============================================================
# NEW v2.0 - MODEL 4: XGBoost + Optuna optimization
# ============================================================
if XGBOOST_AVAILABLE:
    print("\n4. XGBOOST + OPTUNA OPTIMIZATION (NEW)")
    print("-" * 50)
    print("   XGBoost: Gradient boosting → handles non-linearity very well")
    print("   Optuna: Automatic hyperparameter search (Bayesian optimization)")

    if OPTUNA_AVAILABLE:
        def xgb_objective(trial):
            params = {
                'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
                'random_state': 42, 'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=False)
            pred = model.predict(X_test)
            return np.sqrt(mean_squared_error(y_test, pred))

        print("\n   Running Optuna study (30 trials)...")
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(xgb_objective, n_trials=30, show_progress_bar=True)
        best_xgb_params = study_xgb.best_params
        best_xgb_params.update({'random_state': 42, 'n_jobs': -1})
        print(f"\n   Best params found: {best_xgb_params}")
        xgb_model = xgb.XGBRegressor(**best_xgb_params)
    else:
        print("   (Optuna not available - using default hyperparameters)")
        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6,
                                      learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, random_state=42, n_jobs=-1)

    evaluate_model('XGBoost', xgb_model, X_train, X_test, y_train, y_test, use_scaled=False)

# ============================================================
# NEW v2.0 - MODEL 5: LightGBM + Optuna
# ============================================================
if LIGHTGBM_AVAILABLE:
    print("\n5. LIGHTGBM + OPTUNA OPTIMIZATION (NEW)")
    print("-" * 50)
    print("   LightGBM: Faster than XGBoost, excellent for large datasets")

    if OPTUNA_AVAILABLE:
        def lgb_objective(trial):
            params = {
                'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves':       trial.suggest_int('num_leaves', 20, 150),
                'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)])
            pred = model.predict(X_test)
            return np.sqrt(mean_squared_error(y_test, pred))

        print("\n   Running Optuna study (30 trials)...")
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lgb_objective, n_trials=30, show_progress_bar=True)
        best_lgb_params = study_lgb.best_params
        best_lgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
        print(f"\n   Best params found: {best_lgb_params}")
        lgb_model = lgb.LGBMRegressor(**best_lgb_params)
    else:
        lgb_model = lgb.LGBMRegressor(n_estimators=300, max_depth=6,
                                       learning_rate=0.05, num_leaves=63,
                                       random_state=42, n_jobs=-1, verbose=-1)

    evaluate_model('LightGBM', lgb_model, X_train, X_test, y_train, y_test, use_scaled=False)

# ================================================================
# 10. COMPARISON & VISUALIZATION
# ================================================================
print("\n[STEP 9] Model Comparison")
print("="*70)

comparison_df = pd.DataFrame({
    'Model':    list(results.keys()),
    'RMSE (MW)':  [results[m]['RMSE'] for m in results],
    'MAE (MW)':   [results[m]['MAE']  for m in results],
    'R² Score':   [results[m]['R2']   for m in results],
    'Train R²':   [results[m]['Train_R2'] for m in results],
    'Gap':        [results[m]['Gap']  for m in results],
})

print("\n--- MODEL COMPARISON TABLE ---")
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.loc[comparison_df['RMSE (MW)'].idxmin(), 'Model']
print(f"\n BEST MODEL: {best_model_name}")

# --- Plot comparison ---
num_models = len(results)
colors = plt.cm.Set2(np.linspace(0, 1, num_models))

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Model Comparison - v2.0 (with XGBoost & LightGBM)', fontsize=14, fontweight='bold')

# RMSE
axes[0].bar(comparison_df['Model'], comparison_df['RMSE (MW)'], color=colors, edgecolor='black')
axes[0].set_title('RMSE (Lower is Better)'); axes[0].tick_params(axis='x', rotation=20)
for i, v in enumerate(comparison_df['RMSE (MW)']): axes[0].text(i, v+10, f'{v:.0f}', ha='center', fontsize=8)

# MAE
axes[1].bar(comparison_df['Model'], comparison_df['MAE (MW)'], color=colors, edgecolor='black')
axes[1].set_title('MAE (Lower is Better)'); axes[1].tick_params(axis='x', rotation=20)

# R²
axes[2].bar(comparison_df['Model'], comparison_df['R² Score'], color=colors, edgecolor='black')
axes[2].set_title('R² Score (Higher is Better)'); axes[2].set_ylim([0, 1])
axes[2].tick_params(axis='x', rotation=20)
for i, v in enumerate(comparison_df['R² Score']): axes[2].text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=8)

# NEW v2.0 - Train vs Test R² (overfitting visualization)
x = np.arange(num_models)
axes[3].bar(x - 0.2, comparison_df['Train R²'], 0.4, label='Train R²', color='steelblue', edgecolor='black')
axes[3].bar(x + 0.2, comparison_df['R² Score'], 0.4, label='Test R²',  color='tomato',    edgecolor='black')
axes[3].set_title('Train vs Test R² (Gap = Overfitting)')
axes[3].set_xticks(x); axes[3].set_xticklabels(comparison_df['Model'], rotation=20)
axes[3].set_ylim([0, 1]); axes[3].legend()

plt.tight_layout()
plt.savefig('4_model_comparison_v2.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 4_model_comparison_v2.png")
plt.show()

# --- Predictions vs Actual ---
fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
if len(results) == 1: axes = [axes]
fig.suptitle('Predictions vs Actual Values', fontsize=14, fontweight='bold')

for idx, (name, res) in enumerate(results.items()):
    axes[idx].scatter(y_test, res['predictions'], alpha=0.3, s=10)
    mn = min(y_test.min(), res['predictions'].min())
    mx = max(y_test.max(), res['predictions'].max())
    axes[idx].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
    axes[idx].set_title(f"{name}\nR²={res['R2']:.4f}")
    axes[idx].set_xlabel('Actual (MW)'); axes[idx].set_ylabel('Predicted (MW)')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('5_predictions_vs_actual_v2.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 5_predictions_vs_actual_v2.png")
plt.show()

# ================================================================
# 11. FEATURE IMPORTANCE (XGBoost or LightGBM)
# ================================================================
best_tree_model = None
if 'LightGBM' in models:
    best_tree_model = ('LightGBM', models['LightGBM'])
elif 'XGBoost' in models:
    best_tree_model = ('XGBoost', models['XGBoost'])
elif 'Random Forest' in models:
    best_tree_model = ('Random Forest', models['Random Forest'])

if best_tree_model:
    name, model = best_tree_model
    importance = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='steelblue', edgecolor='black')
    plt.title(f'Feature Importance - {name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('7_feature_importance.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 7_feature_importance.png")
    plt.show()

# ================================================================
# 12. FINAL SUMMARY
# ================================================================
print("\n" + "="*70)
print("                  FINAL SUMMARY - v2.0")
print("="*70)
print("\n IMPROVEMENTS ADDED (from expert feedback):")
print("   ✓ Cyclic features (sin/cos) for Hour, Month, DayOfWeek, DayOfYear")
print("   ✓ XGBoost model (handles non-linearity, threshold effects, seasonality)")
print("   ✓ LightGBM model (faster, great for large datasets)")
print("   ✓ Optuna hyperparameter optimization (Bayesian search)")
print("   ✓ Train/Test gap analysis (overfitting detection)")
print("   ✓ Temporal split instead of random shuffle (time series integrity)")
print("   ✓ Feature importance visualization")

print("\n RESULTS:")
print(comparison_df[['Model', 'RMSE (MW)', 'R² Score', 'Gap']].to_string(index=False))

print(f"\n Best model: {best_model_name}")
print("\n NEXT STEPS (Kaggle competitions to explore):")
print("   - House Prices: Advanced Regression Techniques")
print("   - Store Sales - Time Series Forecasting")
print("   - M5 Forecasting - Accuracy")
print("\n" + "="*70)
