#!/bin/bash
# Shared Expert Max Throughput Benchmark Script
# No request_rate limit - tests maximum throughput
#
# Three experiments:
# 1. Shared Expert TP8 (baseline)
# 2. Shared Expert DP (Replicated) + Uniform (no load balancing)
# 3. Shared Expert DP (Replicated) + Waterfill for Prefill, Uniform for Decode

set -e

MODEL_PATH="/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3/"
HOST="0.0.0.0"
PORT=30000
RESULT_DIR="/tmp/shared_expert_benchmark_maxtp"

# Benchmark parameters - NO request_rate for max throughput
NUM_PROMPTS=500
RANDOM_INPUT=1024
RANDOM_OUTPUT=1024

mkdir -p ${RESULT_DIR}

wait_for_server() {
    echo "Waiting for server to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:${PORT}/v1/models 2>/dev/null | grep -q 'DeepSeek-V3'; then
            echo "Server is ready!"
            return 0
        fi
        echo "  Still waiting... ($i/60)"
        sleep 10
    done
    echo "Server failed to start!"
    return 1
}

kill_server() {
    echo "Stopping server..."
    pkill -f "launch_server" 2>/dev/null || true
    sleep 5
}

run_benchmark() {
    local name=$1
    local output_file="${RESULT_DIR}/${name}.jsonl"

    echo "Running benchmark: ${name} (max throughput, no request_rate limit)"
    python3 -m sglang.bench_serving \
        --backend sglang \
        --dataset-name random \
        --num-prompts ${NUM_PROMPTS} \
        --random-input ${RANDOM_INPUT} \
        --random-output ${RANDOM_OUTPUT} \
        --model ${MODEL_PATH} \
        --output-file ${output_file}

    echo "Results saved to: ${output_file}"
}

extract_metrics() {
    local file=$1
    python3 -c "
import json
with open('${file}') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            if 'output_throughput' in d:
                print(f\"  Output Throughput: {d['output_throughput']:.2f} tok/s\")
                print(f\"  Mean E2E Latency:  {d['mean_e2e_latency_ms']:.0f} ms\")
                print(f\"  Mean TPOT:         {d['mean_tpot_ms']:.2f} ms\")
                print(f\"  Mean TTFT:         {d['mean_ttft_ms']:.2f} ms\")
                break
"
}

echo "=========================================="
echo "  MAX THROUGHPUT BENCHMARK (No rate limit)"
echo "=========================================="
echo ""

# ==========================================
# Experiment 1: Shared Expert TP8 (baseline)
# ==========================================
echo "=========================================="
echo "Experiment 1: Shared Expert TP8 (Baseline)"
echo "=========================================="
kill_server

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp1_server.log 2>&1 &

wait_for_server
run_benchmark "exp1_tp8_baseline"

echo ""
echo "Experiment 1 Results:"
extract_metrics "${RESULT_DIR}/exp1_tp8_baseline.jsonl"
echo ""

# ==========================================
# Experiment 2: Shared Expert DP + Uniform
# ==========================================
echo "=========================================="
echo "Experiment 2: Shared Expert DP + Uniform"
echo "  (Replicated weights, no load balancing)"
echo "=========================================="
kill_server

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --enable-shared-expert-balance \
    --shared-expert-balance-mode uniform \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp2_server.log 2>&1 &

wait_for_server
run_benchmark "exp2_dp_uniform"

echo ""
echo "Experiment 2 Results:"
extract_metrics "${RESULT_DIR}/exp2_dp_uniform.jsonl"
echo ""

# ==========================================
# Experiment 3: Shared Expert DP + Waterfill
# ==========================================
echo "=========================================="
echo "Experiment 3: Shared Expert DP + Waterfill"
echo "  (Prefill: Waterfill load balancing)"
echo "  (Decode: Uniform for CUDA Graph)"
echo "=========================================="
kill_server

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --tp 8 \
    --ep 8 \
    --moe-a2a-backend none \
    --enable-shared-expert-balance \
    --shared-expert-balance-mode waterfill \
    --host ${HOST} \
    --port ${PORT} \
    --trust-remote-code \
    > ${RESULT_DIR}/exp3_server.log 2>&1 &

wait_for_server
run_benchmark "exp3_dp_waterfill"

echo ""
echo "Experiment 3 Results:"
extract_metrics "${RESULT_DIR}/exp3_dp_waterfill.jsonl"
echo ""

# ==========================================
# Summary
# ==========================================
kill_server

echo "=========================================="
echo "           FINAL SUMMARY                  "
echo "=========================================="
echo ""

python3 << 'PYEOF'
import json

results = {}
files = [
    ("TP8", "/tmp/shared_expert_benchmark_maxtp/exp1_tp8_baseline.jsonl"),
    ("Uniform", "/tmp/shared_expert_benchmark_maxtp/exp2_dp_uniform.jsonl"),
    ("Waterfill", "/tmp/shared_expert_benchmark_maxtp/exp3_dp_waterfill.jsonl")
]

for name, path in files:
    try:
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if "output_throughput" in d:
                        results[name] = d
                        break
    except Exception as e:
        print(f"Error loading {name}: {e}")

if len(results) == 3:
    print("=" * 70)
    print("   MAX THROUGHPUT BENCHMARK RESULTS (CUDA Graph ON, No rate limit)")
    print("=" * 70)
    print()

    header = f"{'Metric':<25} | {'TP8 Baseline':>14} | {'DP+Uniform':>14} | {'DP+Waterfill':>14}"
    print(header)
    print("-" * 70)

    tp = [results["TP8"]["output_throughput"], results["Uniform"]["output_throughput"], results["Waterfill"]["output_throughput"]]
    tpot = [results["TP8"]["mean_tpot_ms"], results["Uniform"]["mean_tpot_ms"], results["Waterfill"]["mean_tpot_ms"]]
    ttft = [results["TP8"]["mean_ttft_ms"], results["Uniform"]["mean_ttft_ms"], results["Waterfill"]["mean_ttft_ms"]]
    e2e = [results["TP8"]["mean_e2e_latency_ms"], results["Uniform"]["mean_e2e_latency_ms"], results["Waterfill"]["mean_e2e_latency_ms"]]

    print(f"{'Output Throughput (tok/s)':<25} | {tp[0]:>14.2f} | {tp[1]:>14.2f} | {tp[2]:>14.2f}")
    print(f"{'Mean TPOT (ms)':<25} | {tpot[0]:>14.2f} | {tpot[1]:>14.2f} | {tpot[2]:>14.2f}")
    print(f"{'Mean TTFT (ms)':<25} | {ttft[0]:>14.2f} | {ttft[1]:>14.2f} | {ttft[2]:>14.2f}")
    print(f"{'Mean E2E Latency (ms)':<25} | {e2e[0]:>14.0f} | {e2e[1]:>14.0f} | {e2e[2]:>14.0f}")
    print()
    print("=" * 70)
    print("                    RELATIVE PERFORMANCE")
    print("=" * 70)
    print()
    print(f"DP+Uniform vs TP8:   {(tp[1]-tp[0])/tp[0]*100:+.2f}% throughput")
    print(f"DP+Waterfill vs TP8: {(tp[2]-tp[0])/tp[0]*100:+.2f}% throughput")
else:
    print("Some experiments failed to complete")
PYEOF

echo ""
echo "All results saved to: ${RESULT_DIR}/"
echo "=========================================="
