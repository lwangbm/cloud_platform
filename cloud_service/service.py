import logging

import grpc

from api.communication import api_pb2, api_pb2_grpc
from cloud_service.base_health_service import HealthServicer
from cloud_service.base_service import BaseCloudService
from utils.constants import DEVICES, MODEL_TYPES

logger = logging.getLogger(__name__)


def _set_model_update_context_error(context, error_message):
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(error_message)
    logger.info(error_message)
    print(error_message)
    return api_pb2.ModelUpdateResponse()


def _set_model_inference_context_error(context, error_message):
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(error_message)
    logger.info(error_message)
    print(error_message)
    return api_pb2.ModelInferenceResponse()


class BaseService(api_pb2_grpc.CloudServiceServicer, HealthServicer):
    def __init__(self):
        super(BaseService, self).__init__()

    # ToDo: implement this function, in case we need cloud-side inference
    def GetModelInference(self, request, context):
        if api_pb2.DeviceType.Name(request.device_type) in DEVICES and api_pb2.ModelType.Name(request.model_type) in MODEL_TYPES:
            service = BaseCloudService()
            is_attack = service.get_model_inference(request)
            return api_pb2.ModelInferenceResponse(is_attack=is_attack)
        else:
            return _set_model_inference_context_error(
                context,
                "device {} or model type {} is not supported".format(
                    request.model_type, request.device_type
                ),
            )

    def GetUpdatedModel(self, request, context):
        if api_pb2.DeviceType.Name(request.device_type) in DEVICES and api_pb2.ModelType.Name(request.model_type) in MODEL_TYPES:
            service = BaseCloudService()
            response = service.get_updated_model(request)
            return response
        else:
            return _set_model_update_context_error(
                context,
                "device {} or model type {} is not supported".format(
                    request.model_type, request.device_type
                ),
            )

    def GetInitialModel(self, request, context):
        if api_pb2.DeviceType.Name(request.device_type) in DEVICES and api_pb2.ModelType.Name(request.model_type) in MODEL_TYPES:
            service = BaseCloudService()
            response = service.get_initial_model(request)
            return response
        else:
            return _set_model_update_context_error(
                context,
                "device {} or model type {} is not supported".format(
                    request.model_type, request.device_type
                ),
            )
