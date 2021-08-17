from __future__ import division

import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

from api.communication import api_pb2
from cloud_backend.ransomware.algorithms.train import train_init_model
from utils import (obj_to_pickle_string, pickle_string_to_obj,
                   proto_matrix_to_array, proto_vector_to_array)


def plot_single(data, color, **kwargs):
    plt.plot(data, color=color, **kwargs)


# ToDo: implement this for cloud-side model inference
def model_predict(request):
    pass


def model_update(request):
    scaler = preprocessing.StandardScaler()
    y = proto_vector_to_array(request.training_data_y)
    x = proto_matrix_to_array(request.training_data_x)
    x = scaler.fit_transform(x)
    assert request.old_model is not None
    model = pickle_string_to_obj(request.old_model.model_data_string)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    model.fit(train_x, train_y)
    predict = model.predict(test_x)
    precision = metrics.precision_score(test_y, predict, average="binary")
    recall = metrics.recall_score(test_y, predict, average="binary")
    print("precision: %.2f%%, recall: %.2f%%" % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    print("accuracy: %.2f%%" % (100 * accuracy))
    return obj_to_pickle_string(model), obj_to_pickle_string(scaler)


def model_init(request):
    scaler = preprocessing.StandardScaler()
    y = proto_vector_to_array(request.training_data_y)
    x = proto_matrix_to_array(request.training_data_x)
    x = scaler.fit_transform(x)

    model = train_init_model(x, y, api_pb2.ModelType.Name(request.model_type))

    return obj_to_pickle_string(model), obj_to_pickle_string(scaler)
