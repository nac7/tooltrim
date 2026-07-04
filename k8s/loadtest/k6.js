// k6 load test for the tooltrim proxy (industry-standard RPS/latency).
//   k6 run -e URL=http://localhost:8800 -e VUS=50 -e DURATION=60s k8s/loadtest/k6.js
//
// Reports throughput and p50/p90/p99 latency. Token/cost savings come from the
// proxy's /metrics (Prometheus) — scrape it or use loadgen.py for a combined view.
import http from "k6/http";
import { check } from "k6";

const URL = __ENV.URL || "http://localhost:8800";

// One bloated tool result with a buried fact — the shape tooltrim targets.
const records = [];
for (let i = 0; i < 600; i++) {
  records.push({ id: i, status: "ok", amount: i * 3.5, note: "routine entry no useful content" });
}
records[411].note = "refund issued to customer 4417 for amount 250";
const toolOutput = JSON.stringify({ page: 1, total: 600, results: records });

const body = JSON.stringify({
  model: "gpt-4o-mini",
  messages: [
    { role: "user", content: "Which customer got a refund?" },
    { role: "assistant", content: null,
      tool_calls: [{ id: "c1", type: "function", function: { name: "list_orders", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "c1", content: toolOutput },
  ],
});

export const options = {
  vus: Number(__ENV.VUS || 50),
  duration: __ENV.DURATION || "60s",
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(99)<2000"],
  },
};

export default function () {
  const res = http.post(`${URL}/v1/chat/completions`, body, {
    headers: { "Content-Type": "application/json", Authorization: "Bearer test" },
  });
  check(res, { "status 200": (r) => r.status === 200 });
}
