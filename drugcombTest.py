import numpy as np

# ============================================================
# 1. USER INPUT SECTION (Variable = Input)
# ============================================================

print(f"\n--- Synergy Score) ---")
syn_score = ("Enter synergy score:")

print(f"\n--- SENSITIVE CELL MEASUREMENTS (at Dosing Conc) ---")
s0       = float(input("Initial Sensitive Cell Count (s0): "))
lambda_s = float(input(f"Measured Net Growth Rate under drug 1 (lambda_S): "))
mu_s     = float(input(f"Measured Net Growth Rate under drug 2 (mu_S): "))

print(f"\n--- RESISTANT CELL MEASUREMENTS (at Dosing Conc) ---")
r0       = float(input("Initial Resistant Cell Count (r0): "))
lambda_r = float(input(f"Measured Net Growth Rate under drug 1 (lambda_R): "))
mu_r     = float(input(f"Measured Net Growth Rate under drug 2 (mu_R): "))

# Synergy Scale Constant (scales synergy score into the growth penalty)
SYNERGY_SCALE = 0.005 

# ============================================================
# 2. LOGIC & CALCULATIONS
# ============================================================

def calculate_plan():
    best_time = 0.0
    best_X = 0.0
    elimination_possible = False
    
    # X represents the fraction of time spent on Drug 1 (t1 / (t1 + t2))
    # We test 101 points from 0.0 to 1.0
    for X in np.linspace(0, 1, 101):
        # Time-averaged growth rates including synergy penalty
        penalty = SYNERGY_SCALE * syn_score * X * (1 - X)
        
        aS = (X * lambda_s + (1 - X) * mu_s) - penalty
        aR = (X * lambda_r + (1 - X) * mu_r) - penalty
        
        # Elimination Logic: Average growth for both S and R must be negative
        if aS < 0 and aR < 0:
            elimination_possible = True
            return True, X, float('inf')
        
        # Time to Progression (POD) Logic
        # Population stays below s0+r0 until it rebounds. 
        # If aR > aS and aS < 0, there is a delay before R takes over.
        if aR > aS and aS < 0 and aR > 0:
            # Solve for t: s0*exp(aS*t) + r0*exp(aR*t) = s0 + r0
            # Approximation for t_star (Time to reach escape)
            t_pod = (1 / (aR - aS)) * np.log(-s0 * aS / (r0 * aR))
            if t_pod > best_time:
                best_time = t_pod
                best_X = X
        elif aR <= 0 and aS <= 0:
            # If both are non-positive but not both negative enough for "elimination"
            # we treat it as a very long delay.
            if 0 > aR > -1e-9: # stable
                t_pod = 1000.0 
                if t_pod > best_time:
                    best_time = t_pod
                    best_X = X

    return elimination_possible, best_X, best_time

# Execute
elim_poss, opt_x, max_t = calculate_plan()

# ============================================================
# 3. FORMATTED OUTPUT
# ============================================================

print(f"elimination_possible: {elim_poss}")
print(f"optimal_fraction_drug1: {round(opt_x, 2)}")
print(f"optimal_fraction_drug2: {round(1 - opt_x, 2)}")
print(f"max_time_to_POD: {max_t if max_t == float('inf') else round(max_t, 5)}")