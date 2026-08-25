"""
Atmospheric Corrosion Modeling using Constrained Ordinal Logistic Regression (C-OLR)

This script implements the C-OLR model used to classify atmospheric
corrosivity into ordered ISO 9223 categories (C2-C5) from environmental
exposure variables.

The script loads the training and independent testing datasets, applies
the specified preprocessing, fits the constrained ordinal logistic model,
and evaluates its classification performance.

Copyright (c) 2026 Arash Rockey
"""

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from scipy.special import expit  

from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score, f1_score
from tabulate import tabulate





# Data Path
path = "C:/Users/Arash Rockey/Desktop/paper/C-OLR/npj/code/datasets.xlsx"

data1 = pd.read_excel(path, sheet_name="training_dataset")
data2 = pd.read_excel(path, sheet_name="testing_dataset") 

category_map = {
    "C1": "C2",
    "C2": "C2",
    "C3": "C3",
    "C4": "C4",
    "C5": "C5",
    "CX": "C5"
}


# Replace unmapped categories securely without turning others into NaN
data1["CAT"] = data1["CAT"].astype(str).str.strip().replace(category_map)
data2["CAT"] = data2["CAT"].astype(str).str.strip().replace(category_map)


'''
Scale environmental variables using ranges calculated from the
training dataset. The same training ranges are applied to the
independent testing dataset.
'''

scale_cols = ['Atmosphere', 'Temperature', 'Cl', 'SO2']

min_1 = data1[scale_cols].min().values
max_1 = data1[scale_cols].max().values

def minmax(df, columns, min_arr, max_arr, eps=1e-6):
    scaled = df.copy()
    for i, c in enumerate(columns):
        denom = max_arr[i] - min_arr[i]
        if denom < eps:
            scaled[c] = 0.0
        else:
            scaled[c] = (df[c] - min_arr[i]) / denom 
    return scaled 

data1 = minmax(data1, scale_cols, min_1, max_1)
data2 = minmax(data2, scale_cols, min_1, max_1)

# Convert TOW from hours/year to a fraction of a year.
data1["TOW"] = data1["TOW"] / (24 * 365)
data2["TOW"] = data2["TOW"] / (24 * 365)





Columns = ['Atmosphere', 'Temperature', 'TOW', 'Cl', 'SO2']
grade_order = ['C2', 'C3', 'C4', 'C5']
grade_to_int = {g: i for i, g in enumerate(grade_order)}

X_train = data1[Columns].values
y_train = data1['CAT'].values
y_train_idx = np.array([grade_to_int[g] for g in y_train])

n_features = X_train.shape[1]
n_thresholds = len(grade_order) - 1

init_beta = np.ones(n_features) * 0.1
init_theta = np.array([5.0, 10.0, 15.0])
init = np.concatenate([init_beta, init_theta])





def nll(params, lam=1e-5):
    weights = params[:n_features]
    theta = params[n_features:]

    pred = X_train @ weights

    # Evaluate cumulative logistic probabilities at the latent thresholds.
    cum_p = expit(theta[None, :] - pred[:, None])
    
    # Add lower and upper cumulative-probability boundaries.
    F = np.hstack([np.zeros((len(pred), 1)), cum_p, np.ones((len(pred), 1))])
    
    # Calculate the probability of the observed ordinal category.
    # Prevent log(0) during optimization.
    # L2 regularization of the regression coefficients.
    
    p_k = F[np.arange(len(pred)), y_train_idx + 1] - F[np.arange(len(pred)), y_train_idx]
    p_k = np.clip(p_k, 1e-12, 1.0)

    penalty = lam * np.sum(weights ** 2)
    return -np.sum(np.log(p_k)) + penalty




bounds = [(0, None)] * n_features + [(None, None)] * n_thresholds

def threshold_order_constraint(params):
    theta = params[n_features:]
    return np.diff(theta)





cons = [{'type': 'ineq', 'fun': threshold_order_constraint}]

res = minimize(nll, init, bounds=bounds, constraints=cons,
               method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})




params_opt = res.x
weights = params_opt[:n_features]
theta = params_opt[n_features:]




print("\nOptimization message:", res.message)

print("\nC-OLR weights:")
for name, value in zip(Columns, weights):
    print(f"{name}: {value:.2f}")

print("\nLearned latent thresholds:")
for i, value in enumerate(theta, start=1):
    print(f"theta{i}: {value:.2f}")

def score_to_letter(score):
    for idx, th in enumerate(theta):
        if score < th:
            return grade_order[idx]
    return grade_order[-1]





X_test = data2[Columns].values
y_test = data2['CAT'].values

yreal = np.array([grade_to_int[g] for g in y_test])

C_OLR_pred = X_test @ weights
pred_letters_C_OLR = [score_to_letter(score) for score in C_OLR_pred]
C_OLR = np.array([grade_to_int[g] for g in pred_letters_C_OLR])




results = [
    ["Accuracy", accuracy_score(yreal, C_OLR)],
    ["MAE", np.mean(np.abs(yreal - C_OLR))],
    ["Within ±1", np.mean(np.abs(yreal - C_OLR) <= 1)],
    ["Within ±2", np.mean(np.abs(yreal - C_OLR) <= 2)],
    ["QWK", cohen_kappa_score(yreal, C_OLR, weights="quadratic")],
    ["Spearman", spearmanr(yreal, C_OLR)[0]],
    ["Kendall's Tau", kendalltau(yreal, C_OLR)[0]],
    ["Macro F1", f1_score(yreal, C_OLR, average="macro")],
    ["Balanced Acc", balanced_accuracy_score(yreal, C_OLR)]
]

print("\nC-OLR Performance")
print(tabulate(results, headers=["Metric", "C-OLR"], tablefmt="grid", floatfmt=".2f"))
