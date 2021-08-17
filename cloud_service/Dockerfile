FROM python:3.6

ENV TARGET_DIR /
ENV PYTHONUNBUFFERED 0

RUN if [ "$(uname -m)" = "ppc64le" ] || [ "$(uname -m)" = "aarch64" ]; then \
    apt-get -y update && \
    apt-get -y install gfortran libopenblas-dev liblapack-dev && \
    pip install cython 'numpy>=1.13.3'; \
    fi
RUN GRPC_HEALTH_PROBE_VERSION=v0.3.1 && \
    if [ "$(uname -m)" = "ppc64le" ]; then \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-ppc64le; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-arm64; \
    else \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64; \
    fi && \
    chmod +x /bin/grpc_health_probe

ADD ./ ${TARGET_DIR}/
WORKDIR  ${TARGET_DIR}/cloud_service
RUN pip install --no-cache-dir -r requirements.txt

RUN chgrp -R 0 ${TARGET_DIR} \
  && chmod -R g+rwX ${TARGET_DIR}

ENV PYTHONPATH ${TARGET_DIR}:${TARGET_DIR}/api/communication:${TARGET_DIR}/api/health/python

ENTRYPOINT ["python", "main.py"]
