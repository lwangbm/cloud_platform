SAMPLING_FREQ = 200  # sample rate
FRAME = 5
WINDOW_SIZE = FRAME * SAMPLING_FREQ  # 5s
POSITIVE_SUBDIR = "DATAPOSITIVE"
NEGATIVE_SUBDIR = "DATANEGATIVE"
TEST_SUBDIR = "DATARAW"
CLEAN_DATA_TRAIN = "data/clean_data_train.npz"
CLEAN_DATA_TEST = "data/clean_data_test.npz"
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 2
ATTACK_THRESHOLD = 4
colors = ["#0000BD", "#D30000", "g", "#FF7C00"]
CLOUD_PORT = 9996
DEVICES = ["ransomware", "concentrator"]
MODEL_TYPES = [
    "naive_bayes_classifier",
    "knn_classifier",
    "logistic_regression_classifier",
    "mlp_classifier",
    "random_forest_classifier",
    "decision_tree_classifier",
    "gradient_boosting_classifier",
    "svm_classifier",
    "svm_cross_validation",
]
