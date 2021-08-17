# Cloud Platform for anomaly detection


## Project Structure
 - [api](api) folder defines cloud service api with protobuf
 - [cloud_service](cloud_service) folder implements GRPC services, running the service, and exposing 9996 port to serve requests. It also includes Dockerfile to build a docker image for container-based deployment
 - [cloud_backend](cloud_backend) folder implements the service backend functions, such as model training, model inference, etc.
 - [mock_test](mock_test) folder is used to test the cloud-edge functionality by a server-mimic device


## API:

Cloud service exposes 9996 port to serve GRPC requests:

 - GetInitialModel(ModelUpdateRequest) returns (ModelUpdateResponse)
 - GetUpdatedModel(ModelUpdateRequest) returns (ModelUpdateResponse)
 - GetModelInference(ModelInferenceRequest) returns (ModelInferenceResponse)

Notes:
1. `GetInitialModel` asks for an initial detection model, and the `ModelUpdateRequest` is supposed to contain two parts: `model_type` and `new_data`
2. `GetUpdatedModel` asks for an updated detection model, and the `ModelUpdateRequest` is supposed to contain two parts: `old_model` and `new_data`
3. `model_type` should be chosen from the `MODEL_TYPES` set defined in utils/constants.py
4. `GetModelInference` is a reserved API for cloud-side model inference, which is supposed to be implemented in the future

For more detailed API designs, please refer to api/communication/qpi.proto


## Mock Test
