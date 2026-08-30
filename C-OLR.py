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
from tabulate import tabulate

from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from scipy.special import expit  

from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score, f1_score





# Data path and combination of C1/C2 and C5/CX categories.
path = "C:/Users/.../datasets.xlsx"

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

data1["CAT"] = data1["CAT"].astype(str).str.strip().replace(category_map)
data2["CAT"] = data2["CAT"].astype(str).str.strip().replace(category_map)


'''
Scale environmental variables using ranges calculated from the
training dataset. The same training ranges are applied to the
independent testing dataset.
'''

scale_cols = ['Temperature','Cl','SO2','WL']

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

columns = ['Temperature', 'TOW', 'Cl', 'SO2']
atmosphere_train = data1['Atmosphere'].values.astype(int)

grade_order = ['C2', 'C3', 'C4', 'C5']
grade_to_int = {g: i for i, g in enumerate(grade_order)}



# Prepare training features, categories, and ordinal class indices.
X_train = data1[columns].values
y_train = data1['CAT'].values
y_train_idx = np.array([grade_to_int[g] for g in y_train])

# α: Rural, Urban, Industrial, Marine
n_atmosphere = 4

# β: Temperature, TOW, Cl, SO2
n_features = X_train.shape[1]

# θ: Latent thresholds between ordinal classes
n_thresholds = len(grade_order) - 1


# Initialize environmental weights, atmospheric effects, and latent thresholds.
init_beta = np.ones(n_features) * 0.1
init_alpha = np.array([0.1, 0.2, 0.3, 0.4])
init_theta = np.array([5.0, 10.0, 15.0])

init = np.concatenate([init_alpha, init_beta, init_theta])



'''
Define the C-OLR objective function and optimization setup.
The objective minimizes the negative log-likelihood with L2
regularization on the environmental weights. Bounds and constraints
enforce non-negative effects and ordered atmospheric effects and thresholds.
The model is optimized using SLSQP.
'''

def nll(params, lam=1e-5):

    alpha = params[:n_atmosphere]
    weights = params[n_atmosphere:n_atmosphere + n_features]
    theta = params[n_atmosphere + n_features:]
    atmosphere_effect = alpha[atmosphere_train - 1]

    pred = atmosphere_effect + X_train @ weights

    ll = 0.0
    K = len(grade_order)

    for i, grade in enumerate(y_train):

        k = grade_order.index(grade)

        if k == 0:
            p = expit(theta[0] - pred[i])

        elif k == K - 1:
            p = 1.0 - expit(theta[-1] - pred[i])

        else:
            p = (
                expit(theta[k] - pred[i])
                -
                expit(theta[k - 1] - pred[i])
            )

        p = np.clip(p, 1e-12, 1.0)

        ll += np.log(p)

    penalty = lam * np.sum(weights ** 2)

    return -ll + penalty





bounds = ([(0, None)] * n_atmosphere
        + [(0, None)] * n_features
        + [(None, None)] * n_thresholds)



def order_constraint(params):

    alpha = params[:n_atmosphere]

    theta = params[4 + n_features:]

    return np.array([
        alpha[1] - alpha[0],
        alpha[2] - alpha[1],
        alpha[3] - alpha[2],
        theta[1] - theta[0],
        theta[2] - theta[1]
    ])

cons = [{'type': 'ineq','fun': order_constraint}]

res = minimize( nll, init, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol': 1e-12,'maxiter': 1000,'disp': False} )

params_opt = res.x
print("Optimization success:", res.success)
print("\nOptimization message:", res.message)






# Rsults
# --------------------------------------------------------------
print("\n*** C-OLR Fitted Model Parameters ***")
alpha = params_opt[:n_atmosphere]
weights = params_opt[n_atmosphere:n_atmosphere + n_features]
theta = params_opt[n_atmosphere + n_features:]

atmosphere_names = {
    1: "Rural",
    2: "Urban",
    3: "Industrial",
    4: "Marine"
}

print("\nLearned atmospheric effects (\u03B1):")
for i, value in enumerate(alpha, start=1):
    print(f"{atmosphere_names[i]}: \u03B1{i} = {value:.2f}")

print("\nC-OLR environmental weights (\u03B2):")
for column, value in zip(columns, weights):
    print(f"\u03B2_{column} = {value:.1f}")

print("\nLearned latent thresholds (\u03B8):")
for i, value in enumerate(theta, start=1):
    print(f"\u03B8{i} = {value:.1f}")


def score_to_letter(score):
    if score < theta[0]:
        return 'C2'
    if score < theta[1]:
        return 'C3'
    if score < theta[2]:
        return 'C4'
    return 'C5'



# TEST DATA
# --------------------------------------------------------------
grade_to_int = {grade: i for i, grade in enumerate(grade_order)}

X = data2[columns].values
atmosphere_test = data2['Atmosphere'].values.astype(int)

y = data2['CAT'].values
yreal = np.array([grade_to_int[grade] for grade in y])

# C-OLR PREDICTION
# --------------------------------------------------------------
atmosphere_effect_test = alpha[atmosphere_test - 1]

COLR_pred = atmosphere_effect_test + X @ weights

pred_letters_COLR = [score_to_letter(score) for score in COLR_pred]
C_OLR = np.array([    grade_to_int[grade] for grade in pred_letters_COLR])

# PERFORMANCE METRICS
# --------------------------------------------------------------
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
