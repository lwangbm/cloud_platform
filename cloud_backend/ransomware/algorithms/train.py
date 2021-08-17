import time
import warnings, pickle

from cloud_backend.ransomware.algorithms.basic import (decision_tree_classifier,
                                         gradient_boosting_classifier,
                                         knn_classifier,
                                         logistic_regression_classifier,
                                         mlp_classifier,
                                         random_forest_classifier,
                                         svm_classifier, svm_cross_validation, naive_bayes_classifier)
from sklearn import metrics
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


def train_init_model(x, y, model_type):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

    classifier = svm_cross_validation
    if model_type == "naive_bayes_classifier":
        classifier = naive_bayes_classifier
    elif model_type == "knn_classifier":
        classifier = knn_classifier
    elif model_type == "logistic_regression_classifier":
        classifier = logistic_regression_classifier
    elif model_type == "random_forest_classifier":
        classifier = random_forest_classifier
    elif model_type == "decision_tree_classifier":
        classifier = decision_tree_classifier
    elif model_type == "svm_classifier":
        classifier = svm_classifier
    elif model_type == "svm_cross_validation":
        classifier = svm_cross_validation
    elif model_type == "gradient_boosting_classifier":
        classifier = gradient_boosting_classifier
    elif model_type == "mlp_classifier":
        classifier = mlp_classifier

    print("******************* %s ********************" % classifier)
    start_time = time.time()
    model = classifier(train_x, train_y)
    print("training took %fs!" % (time.time() - start_time))
    predict = model.predict(test_x)
    precision = metrics.precision_score(test_y, predict, average="binary")
    recall = metrics.recall_score(test_y, predict, average="binary")
    print("precision: %.2f%%, recall: %.2f%%" % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    print("accuracy: %.2f%%" % (100 * accuracy))

    return model
