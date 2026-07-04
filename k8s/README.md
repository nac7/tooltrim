# tooltrim scalability & load test (Kubernetes + shared Redis)

A horizontal-scale study of the tooltrim compression proxy: many stateless
proxy replicas in front of a shared, content-addressed Redis expand store, with
Prometheus/Grafana measuring **throughput, latency, and tokens/dollars saved**.

## Why it scales this way

Compression is **CPU-bound** — tokenization (tiktoken) plus per-content-type
extraction — and Python's GIL limits a single process to ~one core. So a pod
does not get faster with more threads; you add **replicas**. Because expand refs
are **content hashes**, every replica can resolve any expansion from the shared
Redis with no coordination, so the proxy scales out linearly. This is the whole
architectural argument, and the load test demonstrates it.

## Components

| File | Role |
|---|---|
| `serve.py` | Proxy entrypoint; builds a `RedisStore` + `Metrics` from env. |
| `mock_upstream.py` | Canned LLM upstream so the test isolates the proxy (no spend). |
| `Dockerfile` | Builds one image; proxy and mock share it (different commands). |
| `manifests/` | namespace, redis, mock-upstream, proxy Deployment+Service+HPA. |
| `loadtest/loadgen.py` | Stdlib load generator; reports RPS, p50/p99, tokens & $ saved via `/metrics`. |
| `loadtest/k6.js` | k6 script for standardized RPS/latency. |
| `grafana-dashboard.json` | "tokens & dollars saved" dashboard. |

## Run on a cluster

```bash
# 1. Build + push the image (or `kind load docker-image`).
docker build -t tooltrim-proxy:0.2.1 k8s/

# 2. Deploy.
kubectl apply -f k8s/manifests/namespace.yaml
kubectl apply -f k8s/manifests/redis.yaml
kubectl apply -f k8s/manifests/mock-upstream.yaml
kubectl apply -f k8s/manifests/proxy-deploy.yaml

# 3. Load test (port-forward the Service, then fire).
kubectl -n tooltrim port-forward svc/tooltrim-proxy 8800:8800 &
python k8s/loadtest/loadgen.py --url http://localhost:8800 \
    --requests 5000 --concurrency 100 --price-per-mtok 3.0
#   or:  k6 run -e URL=http://localhost:8800 -e VUS=100 k8s/loadtest/k6.js

# 4. Scaling curve: repeat step 3 while scaling replicas and watch RPS climb.
for r in 1 3 6 12; do
  kubectl -n tooltrim scale deploy/tooltrim-proxy --replicas=$r
  kubectl -n tooltrim rollout status deploy/tooltrim-proxy
  python k8s/loadtest/loadgen.py --url http://localhost:8800 --requests 3000 --concurrency 100
done
```

Import `grafana-dashboard.json` into Grafana (Prometheus datasource) for the
tokens/$ saved view. The HPA scales pods on CPU (the throughput-limiting
resource) from 3 to 20 replicas.

## Run locally (no cluster)

The whole path works on one machine — useful for a smoke test or a laptop demo:

```bash
export PYTHONPATH="$PWD"
PORT=9000 python k8s/mock_upstream.py &                 # mock upstream
TOOLTRIM_UPSTREAM=http://localhost:9000 python k8s/serve.py &   # proxy (no Redis needed)
python k8s/loadtest/loadgen.py --url http://localhost:8800 --requests 300 --concurrency 25
```

## Measured (local single pod, no Redis)

A single proxy process on one machine, mock upstream, 300 requests @ concurrency 25,
each carrying a ~5.5k-token bloated JSON tool result:

| metric | value |
|---|---|
| tokens in → out | 5,500,800 → 60,000 |
| **tokens saved** | **98.9%** |
| throughput | ~6 req/s (one GIL-bound core) |
| cost saved (@ $3 / 1M input tokens) | ~$16 on the run |
| **projected @ 1M requests** | **~$54,000 saved** |

The ~6 req/s per core is the point, not a limitation: throughput is CPU-bound, so
the cluster study scales it out with replicas behind the Service while the shared
Redis keeps expansions globally resolvable. Numbers vary with payload size and
`TOOLTRIM_MAX_TOKENS`; the harness reports whatever your workload actually shows.

> Note: the local numbers above are reproducible with the commands in this README.
> Cluster-scale RPS curves require a Kubernetes cluster and are produced by step 4.
