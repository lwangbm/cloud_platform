#!/bin/bash

set -e

CLOUD_IMG=lwangbm/anomaly_detection_cloud:demo

SCRIPT_ROOT=$(dirname ${BASH_SOURCE})/..
cd ${SCRIPT_ROOT}
echo "cd to ${SCRIPT_ROOT}"

docker psuh ${CLOUD_IMG}

echo -e "\n Docker images pull succeeded\n"
