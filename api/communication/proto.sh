#!/bin/bash
export PROTO_FILE=api/communication/api.proto
python3 -m grpc_tools.protoc -I. --python_out=./ --grpc_python_out=./ "$PROTO_FILE"