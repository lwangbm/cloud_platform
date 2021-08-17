import base64
import logging

from api.communication import api_pb2

logger = logging.getLogger(__name__)


class BaseCloudService(object):
    def __init__(self):
        pass

    def get_updated_model(self, request):
        logger.info("-" * 100 + "\n")
        print("-" * 100 + "\n")
        logger.info("New get_updated_model call\n")
        if request.device_type == "ransomware":
            return self._get_updated_model_ransomware(request)
        # ToDo: change this pat, if different devices have various model updating polices
        elif request.device_type == "concentrator":
            return self._get_updated_model_ransomware(request)

    def get_initial_model(self, request):
        logger.info("-" * 100 + "\n")
        print("-" * 100 + "\n")
        logger.info("New get_initial_model call\n")
        if request.device_type == "ransomware":
            return self._get_initial_model_ransomware(request)
        # ToDo: change this pat, if different devices have various model updating polices
        elif request.device_type == "concentrator":
            return self._get_initial_model_ransomware(request)

    # ToDo: implement this function, in case we need cloud-side inference
    @staticmethod
    def get_model_inference(self, request):
        return False

    @staticmethod
    def _get_updated_model_ransomware(request):
        from cloud_backend.ransomware.main import model_update

        model, scaler = model_update(request)
        new_model = api_pb2.Model(
            model_name=request.model_name, model_data_string=model, scaler_string=scaler
        )
        return api_pb2.ModelUpdateResponse(new_model=new_model)

    @staticmethod
    def _get_initial_model_ransomware(self, request):
        from cloud_backend.ransomware.main import model_init

        model, scaler = model_init(request)
        new_model = api_pb2.Model(
            model_name=request.model_name, model_data_string=model, scaler_string=scaler
        )
        return api_pb2.ModelUpdateResponse(new_model=new_model)

    # ToDo: change this pat, if different devices have various model updating polices
    def _get_new_model_concentrator(self, request):
        pass

    @staticmethod
    def encode(name):
        """Encode the name. Chocolate will check if the name contains hyphens.
        Thus we need to encode it.
        """
        return base64.b64encode(name.encode("utf-8")).decode("utf-8")
