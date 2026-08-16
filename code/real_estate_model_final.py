"""
Real Estate Price Prediction Model
Predicting property prices based on characteristics using machine learning regression.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("LOADING DATA...")
print("=" * 80)

df = pd.read_csv('data/scraped_real_estate_model_features.csv')
print(f"Loaded {len(df)} properties")

df = df[df['price_usd'].notna()].copy()
price_99_5 = df['price_usd'].quantile(0.995)
df = df[df['price_usd'] <= price_99_5]

print(f"After cleaning: {len(df)} properties")
print(f"Price range: ${df['price_usd'].min():,.0f} - ${df['price_usd'].max():,.0f}")

print("\n" + "=" * 80)
print("SELECTING FEATURES...")
print("=" * 80)

numeric_features = [
    'covered_area_sqm', 'effective_area_sqm', 'bedrooms', 'bathrooms',
    'area_per_room_sqm', 'amenity_count', 'parking_spaces', 'expenses_usd',
    'is_apartment', 'is_house', 'is_new_construction', 'is_under_construction',
    'has_balcony', 'has_terrace', 'has_garden', 'has_patio', 'has_pool',
    'has_elevator', 'has_security', 'has_air_conditioning', 'has_heating',
    'has_laundry_room', 'has_storage_room', 'has_gym', 'has_grill',
    'is_furnished', 'is_gated_community', 'is_near_beach', 'is_near_park',
    'is_near_sea', 'is_near_subway', 'pets_allowed', 'mortgage_eligible',
]

categorical_features = [
    'property_type', 'area_bucket', 'room_bucket',
]

df_model = df.copy()
available_numeric = []
for col in numeric_features:
    if col in df_model.columns:
        missing_pct = df_model[col].isnull().sum() / len(df_model) * 100
        if missing_pct < 5:
            available_numeric.append(col)

available_categorical = []
for col in categorical_features:
    if col in df_model.columns:
        missing_pct = df_model[col].isnull().sum() / len(df_model) * 100
        if missing_pct < 5:
            available_categorical.append(col)

print(f"Selected {len(available_numeric)} numeric features")
print(f"Selected {len(available_categorical)} categorical features")

print("\n" + "=" * 80)
print("CLEANING DATA...")
print("=" * 80)

features_to_use = available_numeric + available_categorical
df_model = df_model[features_to_use + ['price_usd']].dropna()

print(f"After removing missing values: {len(df_model)} properties")
print(f"Data retained: {len(df_model)/len(df)*100:.1f}%")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING...")
print("=" * 80)

df_model['price_per_sqm'] = df_model['price_usd'] / (df_model['covered_area_sqm'] + 1)
df_model['has_luxury_amenities'] = (df_model['has_pool'] + df_model['has_gym'] + 
                                     df_model['has_grill'] + df_model['has_security']).astype(int)
df_model['location_score'] = (df_model['is_near_beach'] + df_model['is_near_sea'] + 
                               df_model['is_near_park'] + df_model['is_near_subway']).astype(int)

available_numeric.extend(['price_per_sqm', 'has_luxury_amenities', 'location_score'])

print(f"Created 3 derived features")

print("\n" + "=" * 80)
print("PREPARING DATA...")
print("=" * 80)

X = df_model[available_numeric + available_categorical].copy()
y = df_model['price_usd'].copy()

le_dict = {}
for col in available_categorical:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le
    print(f"Encoded '{col}': {len(le.classes_)} categories")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Features normalized")

print("\n" + "=" * 80)
print("TRAINING MODELS...")
print("=" * 80)

models = {}
predictions = {}

print("\n1. Linear Regression")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)

models['Linear Regression'] = {'MAE': mae_lr, 'RMSE': rmse_lr, 'R2': r2_lr, 'MAPE': mape_lr}
predictions['Linear Regression'] = y_pred_lr

print(f"   R2 = {r2_lr:.4f} | MAE = ${mae_lr:,.0f} | MAPE = {mape_lr:.2%}")

print("\n2. Ridge Regression (with regularization)")
ridge = Ridge(alpha=10000)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge)

models['Ridge Regression'] = {'MAE': mae_ridge, 'RMSE': rmse_ridge, 'R2': r2_ridge, 'MAPE': mape_ridge}
predictions['Ridge Regression'] = y_pred_ridge

print(f"   R2 = {r2_ridge:.4f} | MAE = ${mae_ridge:,.0f} | MAPE = {mape_ridge:.2%}")

print("\n3. Random Forest Regressor")
rf = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, 
                           random_state=42, n_jobs=-1, verbose=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

models['Random Forest'] = {'MAE': mae_rf, 'RMSE': rmse_rf, 'R2': r2_rf, 'MAPE': mape_rf}
predictions['Random Forest'] = y_pred_rf

print(f"   R2 = {r2_rf:.4f} | MAE = ${mae_rf:,.0f} | MAPE = {mape_rf:.2%}")

print("\n4. Gradient Boosting Regressor")
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                               random_state=42, verbose=0)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)

models['Gradient Boosting'] = {'MAE': mae_gb, 'RMSE': rmse_gb, 'R2': r2_gb, 'MAPE': mape_gb}
predictions['Gradient Boosting'] = y_pred_gb

print(f"   R2 = {r2_gb:.4f} | MAE = ${mae_gb:,.0f} | MAPE = {mape_gb:.2%}")

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(models).T
print("\n" + results_df.to_string())

best_model_idx = results_df['R2'].idxmax()
best_r2 = results_df['R2'].max()
print(f"\nBEST MODEL: {best_model_idx}")
print(f"   R2 Score: {best_r2:.4f} (explains {best_r2*100:.1f}% of price variance)")
print(f"   MAE: ${results_df.loc[best_model_idx, 'MAE']:,.0f}")
print(f"   MAPE: {results_df.loc[best_model_idx, 'MAPE']:.2%}")

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (from Random Forest)")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"   {row['Feature']:30s} {row['Importance']:.4f}")

print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS (First 10 Test Samples)")
print("=" * 80)

sample_results = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': predictions[best_model_idx][:10],
})
sample_results['Error'] = abs(sample_results['Actual'] - sample_results['Predicted'])
sample_results['Error %'] = (sample_results['Error'] / sample_results['Actual'] * 100).round(1)

print("\n" + sample_results.to_string(index=False))

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS...")
print("=" * 80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0:2])
ax1.scatter(y_test, predictions[best_model_idx], alpha=0.5, s=30, color='steelblue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Price (USD)', fontsize=11)
ax1.set_ylabel('Predicted Price (USD)', fontsize=11)
ax1.set_title(f'{best_model_idx}: Actual vs Predicted', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
model_names = list(results_df.index)
r2_scores = results_df['R2'].values
colors = ['gold' if name == best_model_idx else 'steelblue' for name in model_names]
ax2.barh(range(len(model_names)), r2_scores, color=colors)
ax2.set_yticks(range(len(model_names)))
ax2.set_yticklabels(model_names)
ax2.set_xlabel('R2 Score', fontsize=10)
ax2.set_title('Model Comparison (R2)', fontsize=11, fontweight='bold')
ax2.set_xlim([0, 1])
ax2.grid(True, alpha=0.3, axis='x')

ax3 = fig.add_subplot(gs[1, 0])
mae_scores = results_df['MAE'].values
ax3.barh(range(len(model_names)), mae_scores, color='coral')
ax3.set_yticks(range(len(model_names)))
ax3.set_yticklabels(model_names)
ax3.set_xlabel('MAE (USD)', fontsize=10)
ax3.set_title('Mean Absolute Error', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

ax4 = fig.add_subplot(gs[1, 1])
mape_scores = results_df['MAPE'].values * 100
ax4.barh(range(len(model_names)), mape_scores, color='lightgreen')
ax4.set_yticks(range(len(model_names)))
ax4.set_yticklabels(model_names)
ax4.set_xlabel('MAPE (%)', fontsize=10)
ax4.set_title('Mean Absolute % Error', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

ax5 = fig.add_subplot(gs[1, 2])
residuals = y_test - predictions[best_model_idx]
ax5.hist(residuals, bins=40, color='mediumpurple', alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Prediction Error (USD)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Residuals Distribution', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, :2])
top_n = 12
top_features = feature_importance.head(top_n)
ax6.barh(range(len(top_features)), top_features['Importance'].values, color='teal')
ax6.set_yticks(range(len(top_features)))
ax6.set_yticklabels(top_features['Feature'].values, fontsize=9)
ax6.set_xlabel('Importance Score', fontsize=10)
ax6.set_title('Top 12 Important Features', fontsize=11, fontweight='bold')
ax6.invert_yaxis()
ax6.grid(True, alpha=0.3, axis='x')

ax7 = fig.add_subplot(gs[2, 2])
ax7.hist(y_test, bins=30, color='skyblue', alpha=0.7, edgecolor='black', label='Actual')
ax7.set_xlabel('Price (USD)', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title('Test Set Price Distribution', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

plt.savefig('output/real_estate_model_results.png', dpi=150, bbox_inches='tight')
print("Saved: output/real_estate_model_results.png")
plt.close()