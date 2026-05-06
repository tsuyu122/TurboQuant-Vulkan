# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
# See LICENSE file for details.
"""
Full benchmark: F16 vs Q4_0 vs TQ3_0 KV cache
Tests at increasing context sizes to find max sustainable context.
Uses 30 GPU layers on Vulkan (AMD RX 6750 XT 12GB).
"""

import json, time, sys, os, subprocess, urllib.request, urllib.error, csv, atexit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER = os.path.join(SCRIPT_DIR, "llama_src", "build_vulkan", "bin", "Release", "llama-server.exe")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "google_gemma-4-26B-A4B-it-Q4_K_M.gguf")
HOST = "127.0.0.1"
PORT = 8095
BASE_URL = f"http://{HOST}:{PORT}"

NGL = 30
THREADS = 8

# KV cache types to benchmark
KV_TYPES = ["f16", "q4_0", "tq3_0"]

# Context sizes to try (ascending) - will stop when server fails to start
CONTEXT_SIZES = [8192, 16384, 32768, 65536, 131072]

# Prompt for benchmarking
PROMPT = (
    "Explain in great detail how a modern CPU works, covering the instruction pipeline, "
    "branch prediction, cache hierarchy (L1/L2/L3), out-of-order execution, speculative "
    "execution, SIMD units, memory controller, and how all these components interact. "
    "Then explain how GPU compute differs from CPU compute, covering SIMT architecture, "
    "warp scheduling, shared memory, register files, and memory coalescing."
)

MAX_TOKENS = 256
STARTUP_TIMEOUT = 180  # seconds to wait for server

server_proc = None
results = []

def kill_servers():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True, timeout=10)
        time.sleep(2)
    except Exception:
        pass

def cleanup():
    global server_proc
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        server_proc = None

atexit.register(cleanup)

def wait_for_server(timeout=STARTUP_TIMEOUT):
    start = time.time()
    for i in range(timeout):
        if server_proc and server_proc.poll() is not None:
            return False
        try:
            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    elapsed = time.time() - start
                    print(f"    Server ready in {elapsed:.1f}s", flush=True)
                    return True
        except Exception:
            pass
        if i % 15 == 0 and i > 0:
            print(f"    ...waiting {time.time()-start:.0f}s", flush=True)
        time.sleep(1)
    return False

def start_server(kv_type, ctx_size):
    global server_proc
    kill_servers()
    time.sleep(1)

    log_file = os.path.join(SCRIPT_DIR, "bench_server.log")

    cmd = [
        LLAMA_SERVER,
        "-m", MODEL_PATH,
        "--host", HOST, "--port", str(PORT),
        "-t", str(THREADS),
        "-ngl", str(NGL),
        "-c", str(ctx_size),
        "-ctk", kv_type, "-ctv", kv_type,
        "-np", "1",
        "--no-warmup",
    ]

    print(f"  Starting server: KV={kv_type} CTX={ctx_size//1024}K NGL={NGL}", flush=True)

    with open(log_file, "w") as lf:
        server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=lf)

    if not wait_for_server():
        print(f"    FAILED to start (VRAM exhausted or crash)", flush=True)
        cleanup()
        try:
            with open(log_file, "r") as lf:
                log = lf.read()
                for line in log.split("\n"):
                    if "error" in line.lower() or "failed" in line.lower() or "abort" in line.lower():
                        print(f"    LOG: {line.strip()}", flush=True)
        except Exception:
            pass
        return False
    return True

def get_server_metrics():
    log_file = os.path.join(SCRIPT_DIR, "bench_server.log")
    kv_mem = "N/A"
    model_mem = "N/A"
    try:
        with open(log_file, "r") as f:
            for line in f:
                if "KV self size" in line:
                    parts = line.split("KV self size")
                    if len(parts) > 1:
                        kv_mem = parts[1].strip().rstrip(",").strip("= ")
                if "model buffer size" in line and "Vulkan" in line:
                    parts = line.split("model buffer size")
                    if len(parts) > 1:
                        model_mem = parts[1].strip().rstrip(",").strip("= ")
    except Exception:
        pass
    return kv_mem, model_mem

def run_inference():
    body = json.dumps({
        "model": "gemma4",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"    Inference failed: {e}", flush=True)
        return None

    usage = data.get("usage", {})
    timings = data.get("timings", {})

    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "pp_speed": timings.get("prompt_per_second", 0),
        "tg_speed": timings.get("predicted_per_second", 0),
        "pp_ms": timings.get("prompt_ms", 0),
        "tg_ms": timings.get("predicted_ms", 0),
    }

def run_warmup():
    body = json.dumps({
        "model": "gemma4",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 16,
        "temperature": 0.1,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
        print("    Warmup done", flush=True)
    except Exception:
        pass

def benchmark_config(kv_type, ctx_size):
    if not start_server(kv_type, ctx_size):
        return None

    kv_mem, model_mem = get_server_metrics()
    run_warmup()
    time.sleep(1)

    # Run 3 inferences, take best TG
    best = None
    for run_i in range(3):
        result = run_inference()
        if result is None:
            break
        print(f"    Run {run_i+1}: PP={result['pp_speed']:.2f} t/s  TG={result['tg_speed']:.2f} t/s", flush=True)
        if best is None or result['tg_speed'] > best['tg_speed']:
            best = result

    cleanup()

    if best:
        best['kv_type'] = kv_type
        best['ctx_size'] = ctx_size
        best['kv_mem'] = kv_mem
        best['model_mem'] = model_mem
        return best
    return None

def main():
    print("=" * 70)
    print("TQ3_0 KV Cache Benchmark - F16 vs Q4_0 vs TQ3_0")
    print(f"Model: Gemma 4 26B Q4_K_M | GPU Layers: {NGL} | GPU: AMD RX 6750 XT 12GB")
    print(f"Max tokens: {MAX_TOKENS} | Runs per config: 3 (best of 3)")
    print("=" * 70)
    print()

    max_ctx = {}

    for kv_type in KV_TYPES:
        print(f"\n{'='*50}")
        print(f"Testing KV type: {kv_type.upper()}")
        print(f"{'='*50}")

        for ctx_size in CONTEXT_SIZES:
            print(f"\n  --- Context: {ctx_size//1024}K ---")
            result = benchmark_config(kv_type, ctx_size)

            if result is None:
                print(f"  {kv_type} FAILED at {ctx_size//1024}K context")
                break

            results.append(result)
            max_ctx[kv_type] = ctx_size
            print(f"  BEST: PP={result['pp_speed']:.2f} t/s | TG={result['tg_speed']:.2f} t/s | KV={result['kv_mem']}")

    # Summary
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'KV Type':<10} {'Context':<10} {'PP (t/s)':<12} {'TG (t/s)':<12} {'KV Memory':<20}")
    print("-" * 70)
    for r in results:
        print(f"{r['kv_type']:<10} {r['ctx_size']//1024}K{'':<7} {r['pp_speed']:<12.2f} {r['tg_speed']:<12.2f} {r['kv_mem']:<20}")

    print(f"\n{'='*70}")
    print("MAX CONTEXT SIZE PER KV TYPE")
    print(f"{'='*70}")
    for kv, ctx in max_ctx.items():
        print(f"  {kv.upper():<10} {ctx//1024}K")

    bpv = {"f16": 16.0, "q4_0": 4.5, "tq3_0": 3.5}
    print(f"\n{'='*70}")
    print("COMPRESSION RATIOS (vs F16)")
    print(f"{'='*70}")
    for kv in KV_TYPES:
        ratio = bpv["f16"] / bpv[kv]
        print(f"  {kv.upper():<10} {bpv[kv]:.1f} bpv  {ratio:.2f}x compression")

    # Save CSV
    csv_path = os.path.join(SCRIPT_DIR, "benchmark_full_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "kv_type", "ctx_size", "pp_speed", "tg_speed",
            "prompt_tokens", "completion_tokens", "pp_ms", "tg_ms",
            "kv_mem", "model_mem"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
