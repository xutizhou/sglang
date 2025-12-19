#!/bin/bash
# Shared Expert Load Balancing Benchmark Script
#
# Three experiments:
# 1. Shared Expert TP8 (baseline)
# 2. Shared Expert DP (Replicated) + Uniform (no load balancing)
# 3. Shared Expert DP (Replicated) + Waterfill for Prefill, Uniform for Decode (CUDA Graph compatible)

set -e

MODEL_PATH="/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3/"
HOST="0.0.0.0"
PORT=30000
RESULT_DIR="/tmp/shared_expert_benchmark"

# Benchmark parameters
NUM_PROMPTS=500
RANDOM_INPUT=1024
RANDOM_OUTPUT=1024
REQUEST_RATE=4

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

    echo "Running benchmark: ${name}"
    python3 -m sglang.bench_serving \
        --backend sglang \
        --dataset-name random \
        --num-prompts ${NUM_PROMPTS} \
        --random-input ${RANDOM_INPUT} \
        --random-output ${RANDOM_OUTPUT} \
        --request-rate ${REQUEST_RATE} \
        --model ${MODEL_PATH} \
        --output-file ${output_file}

    echo "Results saved to: ${output_file}"
}

extract_metrics() {
    local file=$1
    python3 -c "
import json
with open('${file}') as f:
    d = json.load(f)
print(f\"  Output Throughput: {d['output_throughput']:.2f} tok/s\")
print(f\"  Mean E2E Latency: {d['mean_e2e_latency_ms']:.0f} ms\")
print(f\"  Mean TPOT: {d['mean_tpot_ms']:.2f} ms\")
print(f\"  Mean TTFT: {d['mean_ttft_ms']:.2f} ms\")
"
}

echo "=========================================="
echo "Shared Expert Load Balancing Benchmark"
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
# Experiment 3: Shared Expert DP + Waterfill (Prefill) + Uniform (Decode)
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
echo "                SUMMARY                   "
echo "=========================================="
echo ""
echo "Experiment 1 (TP8 Baseline):"
extract_metrics "${RESULT_DIR}/exp1_tp8_baseline.jsonl"
echo ""
echo "Experiment 2 (DP + Uniform):"
extract_metrics "${RESULT_DIR}/exp2_dp_uniform.jsonl"
echo ""
echo "Experiment 3 (DP + Waterfill/Uniform):"
extract_metrics "${RESULT_DIR}/exp3_dp_waterfill.jsonl"
echo ""
echo "All results saved to: ${RESULT_DIR}/"
echo "=========================================="
