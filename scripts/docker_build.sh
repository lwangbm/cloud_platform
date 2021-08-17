#!/bin/bash

set -e


CLOUD_IMG=lwangbm/anomaly_detection_cloud:demo

SCRIPT_ROOT=$(dirname ${BASH_SOURCE})/..
cd ${SCRIPT_ROOT}
echo "cd to ${SCRIPT_ROOT}"

docker build -t ${CLOUD_IMG} -f  cloud_service/Dockerfile .

echo -e "\n Docker images build succeeded\n"
