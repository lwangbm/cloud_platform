from __future__ import print_function

import pickle

import grpc
import numpy as np

from api.communication import api_pb2, api_pb2_grpc
from utils import (np_array_to_proto_matrix, obj_to_pickle_string,
                   pickle_string_to_obj, vector_to_proto_vector)
from utils.constants import CLOUD_PORT

CLEAN_DATA_TEST = "clean_data_train.npz"
cloud_server = "localhost:%d" % CLOUD_PORT
channel_cloud = grpc.insecure_channel(cloud_server)
npzfile = np.load(CLEAN_DATA_TEST)
x = npzfile["x"]
y = npzfile["y"]
x_matrix = np_array_to_proto_matrix(x)
y_vector = vector_to_proto_vector(y)
device_type = api_pb2.RANSOMWARE
model_type = api_pb2.knn_classifier


def test_get_init_model():
    proto_request = api_pb2.ModelUpdateRequest(
        training_data_x=x_matrix,
        training_data_y=y_vector,
        device_type=device_type,
        model_type=model_type,
    )
    stub_ = api_pb2_grpc.CloudServiceStub(channel_cloud)
    result = stub_.GetInitialModel(proto_request, timeout=20)
    print(result)
    new_model = result.new_model
    model_data_string = new_model.model_data_string
    scaler_string = new_model.scaler_string

    model = pickle_string_to_obj(model_data_string)
    scaler = pickle_string_to_obj(scaler_string)
    test_x_transform = scaler.transform(x)
    test_prediction = model.predict(test_x_transform)
    print(test_prediction)


def test_get_updted_model():
    with open("best_model.pickle", "rb") as f:
        model = pickle.load(f)

    old_model = api_pb2.Model(model_data_string=obj_to_pickle_string(model))

    proto_request = api_pb2.ModelUpdateRequest(
        training_data_x=x_matrix,
        training_data_y=y_vector,
        device_type=device_type,
        old_model=old_model,
    )
    stub_ = api_pb2_grpc.CloudServiceStub(channel_cloud)

    result = stub_.GetUpdatedModel(proto_request, timeout=20)
    print(result)
    new_model = result.new_model
    model_data_string = new_model.model_data_string
    scaler_string = new_model.scaler_string

    model = pickle_string_to_obj(model_data_string)
    scaler = pickle_string_to_obj(scaler_string)
    test_x_transform = scaler.transform(x)
    test_prediction = model.predict(test_x_transform)
    print(test_prediction)


def main():
    # test_get_init_model()
    test_get_updted_model()


if __name__ == "__main__":
    main()
