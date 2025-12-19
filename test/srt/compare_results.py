#!/usr/bin/env python3
import json
import sys

results = {}
files = [
    ("TP8", "/tmp/shared_expert_benchmark/exp1_tp8_baseline.jsonl"),
    ("Uniform", "/tmp/shared_expert_benchmark/exp2_dp_uniform.jsonl"),
    ("Waterfill", "/tmp/shared_expert_benchmark/exp3_dp_waterfill.jsonl"),
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
        sys.exit(1)

print("=" * 70)
print("      SHARED EXPERT BENCHMARK RESULTS (CUDA Graph ON)")
print("=" * 70)
print()

header = (
    f"{'Metric':<25} | {'TP8 Baseline':>14} | {'DP+Uniform':>14} | {'DP+Waterfill':>14}"
)
print(header)
print("-" * 70)

tp = [
    results["TP8"]["output_throughput"],
    results["Uniform"]["output_throughput"],
    results["Waterfill"]["output_throughput"],
]
tpot = [
    results["TP8"]["mean_tpot_ms"],
    results["Uniform"]["mean_tpot_ms"],
    results["Waterfill"]["mean_tpot_ms"],
]
ttft = [
    results["TP8"]["mean_ttft_ms"],
    results["Uniform"]["mean_ttft_ms"],
    results["Waterfill"]["mean_ttft_ms"],
]
e2e = [
    results["TP8"]["mean_e2e_latency_ms"],
    results["Uniform"]["mean_e2e_latency_ms"],
    results["Waterfill"]["mean_e2e_latency_ms"],
]

print(
    f"{'Output Throughput (tok/s)':<25} | {tp[0]:>14.2f} | {tp[1]:>14.2f} | {tp[2]:>14.2f}"
)
print(
    f"{'Mean TPOT (ms)':<25} | {tpot[0]:>14.2f} | {tpot[1]:>14.2f} | {tpot[2]:>14.2f}"
)
print(
    f"{'Mean TTFT (ms)':<25} | {ttft[0]:>14.2f} | {ttft[1]:>14.2f} | {ttft[2]:>14.2f}"
)
print(
    f"{'Mean E2E Latency (ms)':<25} | {e2e[0]:>14.0f} | {e2e[1]:>14.0f} | {e2e[2]:>14.0f}"
)
print()
print("=" * 70)
print("                       RELATIVE PERFORMANCE")
print("=" * 70)
print()
print(
    f"DP+Uniform vs TP8:   {(tp[1]-tp[0])/tp[0]*100:+.2f}% throughput ({(tpot[0]-tpot[1])/tpot[0]*100:+.2f}% TPOT)"
)
print(
    f"DP+Waterfill vs TP8: {(tp[2]-tp[0])/tp[0]*100:+.2f}% throughput ({(tpot[0]-tpot[2])/tpot[0]*100:+.2f}% TPOT)"
)
print()
print("Notes:")
print("  - TP8: Shared experts use Tensor Parallelism (each rank has 1/8 weights)")
print("  - DP+Uniform: Shared experts replicated, static round-robin assignment")
print(
    "  - DP+Waterfill: Shared experts replicated, waterfill (prefill) + uniform (decode)"
)
