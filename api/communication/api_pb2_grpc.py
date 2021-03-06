# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from api.communication import api_pb2 as api_dot_communication_dot_api__pb2


class CloudServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetUpdatedModel = channel.unary_unary(
            "/api.communication.CloudService/GetUpdatedModel",
            request_serializer=api_dot_communication_dot_api__pb2.ModelUpdateRequest.SerializeToString,
            response_deserializer=api_dot_communication_dot_api__pb2.ModelUpdateResponse.FromString,
        )
        self.GetModelInference = channel.unary_unary(
            "/api.communication.CloudService/GetModelInference",
            request_serializer=api_dot_communication_dot_api__pb2.ModelInferenceRequest.SerializeToString,
            response_deserializer=api_dot_communication_dot_api__pb2.ModelInferenceResponse.FromString,
        )
        self.GetInitialModel = channel.unary_unary(
            "/api.communication.CloudService/GetInitialModel",
            request_serializer=api_dot_communication_dot_api__pb2.ModelUpdateRequest.SerializeToString,
            response_deserializer=api_dot_communication_dot_api__pb2.ModelUpdateResponse.FromString,
        )


class CloudServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetUpdatedModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def GetModelInference(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def GetInitialModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_CloudServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "GetUpdatedModel": grpc.unary_unary_rpc_method_handler(
            servicer.GetUpdatedModel,
            request_deserializer=api_dot_communication_dot_api__pb2.ModelUpdateRequest.FromString,
            response_serializer=api_dot_communication_dot_api__pb2.ModelUpdateResponse.SerializeToString,
        ),
        "GetModelInference": grpc.unary_unary_rpc_method_handler(
            servicer.GetModelInference,
            request_deserializer=api_dot_communication_dot_api__pb2.ModelInferenceRequest.FromString,
            response_serializer=api_dot_communication_dot_api__pb2.ModelInferenceResponse.SerializeToString,
        ),
        "GetInitialModel": grpc.unary_unary_rpc_method_handler(
            servicer.GetInitialModel,
            request_deserializer=api_dot_communication_dot_api__pb2.ModelUpdateRequest.FromString,
            response_serializer=api_dot_communication_dot_api__pb2.ModelUpdateResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "api.communication.CloudService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class CloudService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetUpdatedModel(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/api.communication.CloudService/GetUpdatedModel",
            api_dot_communication_dot_api__pb2.ModelUpdateRequest.SerializeToString,
            api_dot_communication_dot_api__pb2.ModelUpdateResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def GetModelInference(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/api.communication.CloudService/GetModelInference",
            api_dot_communication_dot_api__pb2.ModelInferenceRequest.SerializeToString,
            api_dot_communication_dot_api__pb2.ModelInferenceResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def GetInitialModel(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/api.communication.CloudService/GetInitialModel",
            api_dot_communication_dot_api__pb2.ModelUpdateRequest.SerializeToString,
            api_dot_communication_dot_api__pb2.ModelUpdateResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
