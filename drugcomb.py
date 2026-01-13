import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from category_encoders import TargetEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

# ===========================
# 1. LOAD DATA
# ===========================
print("1. Loading Data...")
train_df = pd.read_csv("ch1_train_combination_and_monoTherapy.csv")
leaderboard_df = pd.read_csv("ch1_leaderBoard_monoTherapy.csv")

# Load OMICS MATRICES
print("   Loading Mutation...")
mutation_df = pd.read_csv("filtered_mutation_matrix.csv")
print("   Loading CNV...")
cnv_df = pd.read_csv("filtered_cnv_matrix.csv")
print("   Loading Methylation...")
meth_df = pd.read_csv("filtered_methylation_matrix.csv")
print("   Loading Gene Expression...")
gex_df = pd.read_csv("filtered_gex_matrix.csv")

# ===========================
# 2. MERGE OMICS (MUT + CNV + METH + GEX)
# ===========================
print("2. Preparing Genetic Data...")

# A. Standardize Cell Names
for df in [mutation_df, cnv_df, meth_df, gex_df]:
    df["cell_line_name"] = df["cell_line_name"].str.upper().str.strip()

# B. Merge Mutation + CNV (Base)
genetics_df = pd.merge(mutation_df, cnv_df, on="cell_line_name", how="outer").fillna(0)

# C. Merge Methylation (Left Join)
genetics_df = pd.merge(genetics_df, meth_df, on="cell_line_name", how="left")

# D. Merge GEX (Left Join)
genetics_df = pd.merge(genetics_df, gex_df, on="cell_line_name", how="left")

# E. Handle Missing Data (Imputation)
print("   Imputing missing values for Methylation & GEX...")
meth_cols = [c for c in genetics_df.columns if str(c).endswith("_METH")]
gex_cols = [c for c in genetics_df.columns if str(c).endswith("_GEX")]

genetics_df[meth_cols] = genetics_df[meth_cols].fillna(genetics_df[meth_cols].mean()).fillna(0.5)
genetics_df[gex_cols] = genetics_df[gex_cols].fillna(genetics_df[gex_cols].mean()).fillna(0)

print(f"   Combined Omics Shape: {genetics_df.shape}")

# ===========================
# 3. MERGE WITH TRAIN/LEADERBOARD
# ===========================
print("3. Merging with Clinical Data...")

train_df["CELL_LINE"] = train_df["CELL_LINE"].str.upper().str.strip()
leaderboard_df["CELL_LINE"] = leaderboard_df["CELL_LINE"].str.upper().str.strip()

train_merged = pd.merge(train_df, genetics_df, left_on="CELL_LINE", right_on="cell_line_name", how="inner")
leaderboard_merged = pd.merge(leaderboard_df, genetics_df, left_on="CELL_LINE", right_on="cell_line_name", how="inner")

print(f"Training shape after merge: {train_merged.shape}")
genetic_features = [col for col in genetics_df.columns if col != "cell_line_name"]

# ===========================
# 4. CLEAN & FEATURE ENGINEERING
# ===========================
train_clean = train_merged[train_merged["QA"] != -1].copy()
train_clean = train_clean[
    (train_clean["SYNERGY_SCORE"] > -100) &
    (train_clean["SYNERGY_SCORE"] < 150)
]

target = "SYNERGY_SCORE"
categorical_cols = ["COMPOUND_A", "COMPOUND_B"] 
numeric_features = ["MAX_CONC_A", "MAX_CONC_B", "IC50_A", "H_A", "Einf_A", "IC50_B", "H_B", "Einf_B"]

def add_interactions(df):
    df = df.copy()
    df["IC50_A_log"] = np.log1p(df["IC50_A"])
    df["IC50_B_log"] = np.log1p(df["IC50_B"])
    return df

train_clean = add_interactions(train_clean)
leaderboard_merged = add_interactions(leaderboard_merged)

# 1. Create the label based on your threshold (>= 20)
synergy_counts = train_clean["SYNERGY_SCORE"].apply(lambda x: "Synergy" if x >= 20 else "No Synergy").value_counts()

'''
# 2. Plot using Matplotlib
plt.figure(figsize=(6, 5))
bars = plt.bar(synergy_counts.index, synergy_counts.values, color=['lightcoral', 'skyblue'])

# 3. Add formatting (Labels, Title, Count above bars)
plt.title("Count of Synergy vs. No Synergy (Threshold ≥ 20)", fontsize=14)
plt.xlabel("Classification", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)


plt.show()

plt.figure(figsize=(8, 5))
plt.hist(train_clean[target], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
plt.axvline(0, color='black', linestyle='-') # 0 is usually the neutral point
plt.title("Distribution of Synergy Scores (Filtered)")
plt.xlabel("Synergy Score")
plt.ylabel("Frequency")
plt.show()
'''
# ===========================
# 5. FEATURE SELECTION (SMART "FORCE-KEEP")
# ===========================
print("4. Feature Selection...")

# 1. Define Feature Groups
clinical_features = numeric_features + ["IC50_A_log", "IC50_B_log"] + categorical_cols 
available_genetic_features = [c for c in genetic_features if c in train_clean.columns]

print(f"   Clinical Features (Kept): {len(clinical_features)}")
print(f"   Genetic Features (To Filter): {len(available_genetic_features)}")

# 2. Variance Threshold (Applied ONLY to Genetics)
print("   Filtering low-variance genes...")
X_genetics = train_clean[available_genetic_features]
selector_var = VarianceThreshold(threshold=0.05) 
X_genetics_reduced = selector_var.fit_transform(X_genetics)

# Recover names
kept_indices_var = selector_var.get_support(indices=True)
genes_after_var = [available_genetic_features[i] for i in kept_indices_var]
print(f"   Genes after Variance Filter: {len(genes_after_var)}")

# 3. SelectKBest (Applied ONLY to Genetics)
print("   Selecting Top 3,000 Genes based on correlation...")
y = train_clean[target]
selector_kbest = SelectKBest(score_func=f_regression, k=3000)
X_genetics_final = selector_kbest.fit_transform(X_genetics_reduced, y)

# Recover names of the "Best" Genes
kept_indices_kbest = selector_kbest.get_support(indices=True)
best_genes = [genes_after_var[i] for i in kept_indices_kbest]
print(f"   Final Selected Genes: {len(best_genes)}")

# 4. Combine Clinical + Best Genes
final_features = clinical_features + best_genes

# 5. Prepare Final Training Data
X = train_clean[final_features]
leaderboard_reduced = leaderboard_merged[final_features]

# 6. Split & Encode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("   Encoding categorical features...")
encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
X_train_enc = encoder.fit_transform(X_train, y_train)
X_test_enc = encoder.transform(X_test)
X_lb_enc = encoder.transform(leaderboard_reduced)

print(f"   Final Training Shape: {X_train_enc.shape}")

# ===========================
# 6. TRAIN XGBOOST
# ===========================
print("5. Training XGBoost...")

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000, 
    learning_rate=0.01,
    max_depth=7,             
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=3,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_enc, y_train) # Note: Fixed variable name to X_train_enc

# ===========================
# 7. RESULTS
# ===========================
print("6. Evaluating...")
y_pred = model.predict(X_test_enc)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- FINAL RESULTS (MUT + CNV + METH + GEX) ---")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Predict Leaderboard
lb_preds = model.predict(X_lb_enc)
cols_to_keep = [
    "CELL_LINE",
    "COMPOUND_A", "COMPOUND_B",
    "MAX_CONC_A", "MAX_CONC_B",
    "IC50_A", "H_A", "Einf_A",
    "IC50_B", "H_B", "Einf_B"
]

output = leaderboard_merged[cols_to_keep].copy()
output["PREDICTED_SYNERGY"] = lb_preds
output.rename(columns={"COMPOUND_A": "DRUG_1", "COMPOUND_B": "DRUG_2"}, inplace=True)
output.to_csv("leaderboard_predictions_FULL_OMICS.csv", index=False)
print("Saved to leaderboard_predictions_FULL_OMICS.csv")

import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# 8. VISUALIZATION (Predicted vs Actual)
# ===========================

# Set style for a clean look
sns.set_style("whitegrid")
'''
plt.figure(figsize=(10, 8))

# 1. Scatter plot of the Test Data
# Alpha controls transparency (helps see density if points overlap)
plt.scatter(y_test, y_pred, alpha=0.6, color='#4C72B0', edgecolor='k', s=40, label='Test Samples')

# 2. Draw the "Perfect Fit" Diagonal Line (y = x)
# If predictions fall exactly on this line, R2 is 1.0
min_val = min(y_test.min(), y_pred.min()) - 5
max_val = max(y_test.max(), y_pred.max()) + 5
plt.plot([min_val, max_val], [min_val, max_val], color='#C44E52', linestyle='--', linewidth=3, label='Perfect Prediction')

# 3. Add Annotation Box with R2 and RMSE
# This puts the metrics directly on the chart
text_str = f'$R^2 = {r2:.4f}$\n$RMSE = {rmse:.4f}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

# Place text in top-left corner (0.05, 0.95 relative coordinates)
plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=props)

# 4. Labels and Title
plt.xlabel("Actual Synergy Score", fontsize=12, fontweight='bold')
plt.ylabel("Predicted Synergy Score", fontsize=12, fontweight='bold')
plt.title(f"XGBoost Regression Performance\n(Features: Clinical + Top 3000 Genes)", fontsize=15)
plt.legend(loc='lower right', fontsize=12)

# 5. Fix axes to make them square (optional, but helps visual comparison)
plt.axis('equal')
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.tight_layout()
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()



from sklearn.model_selection import RandomizedSearchCV
print("7. Starting Hyperparameter Tuning (Lite Mode)...")

# A. Define the grid (Same as before)
param_grid = {
    'n_estimators': [1000, 1500, 2000],      
    'max_depth': [5, 6, 7],                  
    'learning_rate': [0.01, 0.02],    
    'subsample': [0.7, 0.8],            
    'colsample_bytree': [0.5, 0.6],     
    'min_child_weight': [3, 5]            
}

# B. Initialize the base model (limit threads here too)
xgb_base = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_jobs=2,  # Limit XGBoost itself to 2 threads
    random_state=42
)

# C. Run Random Search
# - n_iter=10: Only try 10 random combos (down from 20)
# - cv=2: Only split data twice (down from 3)
# - n_jobs=2: Only use 2 CPU cores for the search (down from all cores)
search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=10,              
    scoring='neg_root_mean_squared_error',
    cv=2,                   
    verbose=1,
    random_state=42,
    n_jobs=2  # Keep this low to save CPU
)

# D. Fit
print("   Running 20 fits (10 candidates x 2 folds)...")
search.fit(X_train_enc, y_train)

# E. Results
best_model = search.best_estimator_
print(f"\nBest Parameters found: {search.best_params_}")

# ===========================
# CORRECTED FEATURE IMPORTANCE CHECK
# ===========================
import matplotlib.pyplot as plt

# 1. Re-create the exact list of feature names used in training
# (Clinical features FIRST, then the Best Genes)
feature_names_in_order = clinical_features + best_genes

# 2. Get importance from the model
importance = model.feature_importances_

# 3. Create DataFrame
feat_imp = pd.DataFrame({
    'Feature': feature_names_in_order,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# 4. Print & Plot
print("\n--- TRUE TOP 5 PREDICTORS ---")
print(feat_imp.head(5))

plt.figure(figsize=(10, 8))
plt.barh(feat_imp['Feature'].head(5)[::-1], feat_imp['Importance'].head(5)[::-1], color='lightblue')
plt.xlabel("XGBoost Importance Score")
plt.title("Top 5 Drivers")
plt.show()

# Check where the Clinical Features specifically ranked
clinical_cols = ["MAX_CONC_A", "IC50_A", "IC50_A_log", "COMPOUND_A", "COMPOUND_B"]
print("\n--- CLINICAL FEATURE RANKS ---")
for col in clinical_cols:
    if col in feat_imp['Feature'].values:
        row = feat_imp[feat_imp['Feature'] == col].iloc[0]
        # Find the rank (index + 1)
        rank = feat_imp.index.get_loc(row.name) + 1
        print(f"{col}: Rank {rank} (Importance: {row['Importance']:.6f})")
    else:
        print(f"{col}: NOT FOUND (Check feature names)")
'''