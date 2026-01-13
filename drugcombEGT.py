import numpy as np
import pandas as pd

# ============================================================
# LOAD & FILTER SYNERGY PREDICTIONS
# ============================================================

CSV_PATH = "leaderboard_predictions_FULL_OMICS.csv"
SYNERGY_THRESHOLD = 20

df_synergy = pd.read_csv(CSV_PATH)

# Keep only strong synergies (>= 20)
df_synergy = df_synergy[df_synergy["PREDICTED_SYNERGY"] >= SYNERGY_THRESHOLD]

print(f"Loaded {len(df_synergy)} synergistic drug pairs (synergy >= {SYNERGY_THRESHOLD})")

# ============================================================
# 0. GLOBAL SETTINGS
# ============================================================

# How strongly synergy penalizes growth when both drugs are used
SYNERGY_SCALE = 0.005

# Initial population sizes
S0 = 9.99e5        # sensitive cells
R0 = 9.99e3          # resistant cells (1%)

# Fraction of time grid for scheduling
X_GRID = np.linspace(0, 1, 101)

# Base max growth rate (Assumed global for the cell line if not in CSV)
G_MAX_DEFAULT = 0.05 

# ============================================================
# 1. HILL FUNCTION
# ============================================================

def hill_function(C, g_max, g_min, IC50, n):
    if C <= 0:
        return g_max
    return g_min + (g_max - g_min) / (1 + (C / IC50) ** n)

# ============================================================
# 2. DRUG-SPECIFIC GROWTH RATES (Modified to accept params)
# ============================================================

def lambda_S(c1, p):
    # p is the 'current_params' dictionary
    d = p["S"]["d1"]
    return hill_function(c1, p["S"]["g_max"], d["g_min"], d["IC50"], d["n"])

def lambda_R(c1, p):
    d = p["R"]["d1"]
    return hill_function(c1, p["R"]["g_max"], d["g_min"], d["IC50"], d["n"])

def mu_S(c2, p):
    d = p["S"]["d2"]
    return hill_function(c2, p["S"]["g_max"], d["g_min"], d["IC50"], d["n"])

def mu_R(c2, p):
    d = p["R"]["d2"]
    return hill_function(c2, p["R"]["g_max"], d["g_min"], d["IC50"], d["n"])


# ============================================================
# 3. EFFECTIVE (TIME-AVERAGED) GROWTH RATES
# ============================================================

def synergy_penalty(X, synergy_score):
    return SYNERGY_SCALE * synergy_score * X * (1 - X)

def alpha_S(X, c1, c2, synergy_score, p):
    base = X * lambda_S(c1, p) + (1 - X) * mu_S(c2, p)
    return base - synergy_penalty(X, synergy_score)

def alpha_R(X, c1, c2, synergy_score, p):
    base = X * lambda_R(c1, p) + (1 - X) * mu_R(c2, p)
    return base - synergy_penalty(X, synergy_score)


# ============================================================
# 4. TIME TO PROGRESSION OF DISEASE (POD)
# ============================================================

def time_to_POD(s0, r0, X, c1, c2, synergy_score, p):
    aS = alpha_S(X, c1, c2, synergy_score, p)
    aR = alpha_R(X, c1, c2, synergy_score, p)

    # Immediate progression
    if aS >= 0 and aR >= 0:
        return 0.0

    # Elimination
    if aS < 0 and aR < 0:
        return np.inf

    # Competitive release case
    if aR <= aS:
        return 0.0

    numerator = -s0 * aS
    denominator = r0 * aR

    if numerator <= 0 or denominator <= 0:
        return 0.0

    t_star = (1 / (aR - aS)) * np.log(numerator / denominator)

    return max(t_star, 0.0)


# ============================================================
# 5. OPTIMIZE SCHEDULE FOR ONE DRUG PAIR
# ============================================================

def optimize_schedule(c1, c2, synergy_score, current_params):
    best_time = 0.0
    best_X = None

    for X in X_GRID:
        aS = alpha_S(X, c1, c2, synergy_score, current_params)
        aR = alpha_R(X, c1, c2, synergy_score, current_params)

        # Elimination condition
        if aS < 0 and aR < 0:
            return {
                "elimination_possible": True,
                "optimal_fraction_drug1": X,
                "optimal_fraction_drug2": 1 - X,
                "max_time_to_POD": np.inf
            }

        t_pod = time_to_POD(S0, R0, X, c1, c2, synergy_score, current_params)

        if t_pod > best_time:
            best_time = t_pod
            best_X = X

    return {
        "elimination_possible": False,
        "optimal_fraction_drug1": best_X,
        "optimal_fraction_drug2": 1 - best_X if best_X is not None else None,
        "max_time_to_POD": best_time
    }

# ============================================================
# 6. MAIN LOOP (Now uses DataFrame rows)
# ============================================================

results = []

print("Running optimization on filtered pairs...")

for _, row in df_synergy.iterrows():
    
    # 1. Extract Sensitive Params from CSV
    # We convert Einf (0-100) to a g_min growth rate. 
    # If Einf=100 (viability), growth is g_max. If Einf=0, growth is 0.
    g_min_A = G_MAX_DEFAULT * (row['Einf_A'] / 100.0)
    g_min_B = G_MAX_DEFAULT * (row['Einf_B'] / 100.0)

    # 2. Construct the params dictionary dynamically for this row
    current_params = {
        "S": {
            "g_max": G_MAX_DEFAULT,
            "d1": {"IC50": row['IC50_A'], "g_min": g_min_A, "n": row['H_A']},
            "d2": {"IC50": row['IC50_B'], "g_min": g_min_B, "n": row['H_B']},
        },
        "R": {
            # Since CSV has no Resistant data, we simulate it by 
            # multiplying IC50 by 50 and keeping other stats same.
            "g_max": G_MAX_DEFAULT * 0.8, # Resistant often grow slower
            "d1": {"IC50": row['IC50_A'] * 50, "g_min": g_min_A, "n": row['H_A']},
            "d2": {"IC50": row['IC50_B'] * 50, "g_min": g_min_B, "n": row['H_B']},
        }
    }

    # 3. Run Optimization
    res = optimize_schedule(
        c1=row['MAX_CONC_A'],  # Use actual max conc from CSV
        c2=row['MAX_CONC_B'],
        synergy_score=row['PREDICTED_SYNERGY'],
        current_params=current_params
    )

    results.append({
        "cell_line": row["CELL_LINE"],
        "drug1": row["DRUG_1"],
        "drug2": row["DRUG_2"],
        "synergy_score": row["PREDICTED_SYNERGY"],
        **res
    })

# ============================================================
# 7. PRINT RESULTS
# ============================================================

df_results = pd.DataFrame(results)

# Convert synergy_score column to numeric, in case it was object
df_results['synergy_score'] = pd.to_numeric(df_results['synergy_score'], errors='coerce')

# Drop rows with NaN synergy_score just in case
df_results = df_results.dropna(subset=['synergy_score'])

# Sort descending by synergy_score
df_results = df_results.sort_values(by='synergy_score', ascending=False)

# Take top 10
df_top10 = df_results.head(10)

# Print nicely
for _, row in df_top10.iterrows():
    print("\n--- Drug Pair ---")
    for k, v in row.items():
        print(f"{k}: {v}")