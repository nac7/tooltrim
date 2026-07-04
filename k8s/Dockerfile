# tooltrim compression proxy — horizontally scalable, shared-Redis expand store.
FROM python:3.12-slim

WORKDIR /app

# Install tooltrim with the redis extra (shared expand store) from PyPI.
# Pin to the released version for reproducible load-test numbers.
ARG TOOLTRIM_VERSION=0.2.1
RUN pip install --no-cache-dir "tooltrim[redis,tokens]==${TOOLTRIM_VERSION}"

# Entrypoints (proxy + mock upstream for isolated load tests).
COPY serve.py mock_upstream.py /app/

EXPOSE 8800
ENV TOOLTRIM_HOST=0.0.0.0 TOOLTRIM_PORT=8800 TOOLTRIM_MAX_TOKENS=512

# tiktoken + compression is CPU-bound and GIL-limited to ~one core per process,
# so scale throughput by adding replicas (see manifests/proxy-deploy.yaml), not
# threads. Keep each pod to ~1 CPU and let the HPA add pods under load.
CMD ["python", "serve.py"]
