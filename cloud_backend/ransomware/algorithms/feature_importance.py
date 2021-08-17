import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from cloud_backend.ransomware.algorithms.basic import (
    decision_tree_classifier, gradient_boosting_classifier, knn_classifier,
    logistic_regression_classifier, mlp_classifier, random_forest_classifier,
    svm_classifier, svm_cross_validation)
from utils.constants import CLEAN_DATA_TRAIN

warnings.filterwarnings("ignore")

npzfile = np.load(os.path.join("..", CLEAN_DATA_TRAIN))
x = npzfile["x_original"]
y = npzfile["y"]
best_accuracy = 0
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
feature_names = [
    "up4",
    "down4",
    "max",
    "min",
    "max-min",
    "down4 - up4",
    "var",
    "std/avg",
    "skew",
    "kurtosis",
    "fft_max",
    "fft_min",
    "std",
    "avg",
    "median",
]

#  model
model = random_forest_classifier(train_x, train_y)

# importance basic
start_time = time.time()
importance = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

# importance using permutation
result = permutation_importance(
    model, test_x, test_y, n_repeats=10, random_state=42, n_jobs=2
)
forest_importance = pd.Series(result.importances_mean, index=feature_names)
fig, ax = plt.subplots()
forest_importance.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importance using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.savefig(
    "./ranking_results/Feature importance using permutation" + ".pdf",
    format="pdf",
    dpi=2400,
    transparent=True,
)

# importance using MDI
forest_importance = pd.Series(importance, index=feature_names)
fig, ax = plt.subplots()
forest_importance.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importance using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig(
    "./ranking_results/Feature importance using MDI" + ".pdf",
    format="pdf",
    dpi=2400,
    transparent=True,
)
