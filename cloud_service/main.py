import time
from concurrent import futures

import grpc

from api.communication import api_pb2_grpc
from api.health.python import health_pb2_grpc
from cloud_service.service import BaseService
from utils.constants import CLOUD_PORT

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
DEFAULT_PORT = "0.0.0.0:%d" % CLOUD_PORT


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = BaseService()
    api_pb2_grpc.add_CloudServiceServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(service, server)
    server.add_insecure_port(DEFAULT_PORT)
    print("Listening...")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
