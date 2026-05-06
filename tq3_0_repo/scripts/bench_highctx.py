# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
# See LICENSE file for details.
"""
High-context benchmark: test 256K, 512K, 1M+ for F16/Q4_0/TQ3_0.
Finds the maximum context each KV type can handle.
Uses 30 GPU layers on Vulkan (AMD RX 6750 XT 12GB).
"""

import json, time, sys, os, subprocess, urllib.request, urllib.error, atexit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER = os.path.join(SCRIPT_DIR, "llama_src", "build_vulkan", "bin", "Release", "llama-server.exe")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "google_gemma-4-26B-A4B-it-Q4_K_M.gguf")
HOST = "127.0.0.1"
PORT = 8095
BASE_URL = f"http://{HOST}:{PORT}"

NGL = 30
THREADS = 8
MAX_TOKENS = 128
STARTUP_TIMEOUT = 300  # longer timeout for huge contexts

# Test 256K, 512K, 1M, 2M
CONTEXT_SIZES = [262144, 524288, 1048576, 2097152]

KV_TYPES = ["f16", "q4_0", "tq3_0"]

PROMPT = (
    "Explain in great detail how a modern CPU works, covering the instruction pipeline, "
    "branch prediction, cache hierarchy (L1/L2/L3), out-of-order execution, speculative "
    "execution, SIMD units, memory controller, and how all these components interact. "
    "Then explain how GPU compute differs from CPU compute, covering SIMT architecture, "
    "warp scheduling, shared memory, register files, and memory coalescing."
)

server_proc = None

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
        if i % 30 == 0 and i > 0:
            print(f"    ...waiting {time.time()-start:.0f}s", flush=True)
        time.sleep(1)
    return False

def start_server(kv_type, ctx_size):
    global server_proc
    kill_servers()
    time.sleep(1)

    log_file = os.path.join(SCRIPT_DIR, "bench_highctx.log")
    ctx_label = f"{ctx_size//1024}K" if ctx_size < 1048576 else f"{ctx_size//1048576}M"

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

    print(f"  Starting server: KV={kv_type} CTX={ctx_label} NGL={NGL}", flush=True)

    with open(log_file, "w") as lf:
        server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=lf)

    if not wait_for_server():
        print(f"    FAILED to start", flush=True)
        cleanup()
        try:
            with open(log_file, "r") as lf:
                log = lf.read()
                for line in log.split("\n")[-20:]:
                    if any(w in line.lower() for w in ["error", "fail", "abort", "alloc", "memory"]):
                        print(f"    LOG: {line.strip()}", flush=True)
        except Exception:
            pass
        return False
    return True

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
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"    Inference failed: {e}", flush=True)
        return None

    timings = data.get("timings", {})
    return {
        "pp_speed": timings.get("prompt_per_second", 0),
        "tg_speed": timings.get("predicted_per_second", 0),
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

def fmt_ctx(ctx_size):
    if ctx_size >= 1048576:
        return f"{ctx_size//1048576}M"
    return f"{ctx_size//1024}K"

def main():
    print("=" * 70)
    print("HIGH CONTEXT BENCHMARK - F16 vs Q4_0 vs TQ3_0")
    print(f"Model: Gemma 4 26B Q4_K_M | GPU Layers: {NGL} | GPU: AMD RX 6750 XT 12GB")
    print(f"Context sizes: {', '.join(fmt_ctx(c) for c in CONTEXT_SIZES)}")
    print(f"Max tokens: {MAX_TOKENS} | Runs: 2 (best of 2)")
    print("=" * 70)

    results = []
    max_ctx = {}

    for kv_type in KV_TYPES:
        print(f"\n{'='*50}")
        print(f"Testing KV type: {kv_type.upper()}")
        print(f"{'='*50}")

        for ctx_size in CONTEXT_SIZES:
            label = fmt_ctx(ctx_size)
            print(f"\n  --- Context: {label} ---")

            if not start_server(kv_type, ctx_size):
                print(f"  {kv_type.upper()} CANNOT START at {label} — max is {fmt_ctx(max_ctx.get(kv_type, 0))}")
                break

            run_warmup()
            time.sleep(1)

            best = None
            for run_i in range(2):
                result = run_inference()
                if result is None:
                    break
                print(f"    Run {run_i+1}: PP={result['pp_speed']:.2f} t/s  TG={result['tg_speed']:.2f} t/s", flush=True)
                if best is None or result['tg_speed'] > best['tg_speed']:
                    best = result

            cleanup()

            if best:
                max_ctx[kv_type] = ctx_size
                results.append({"kv_type": kv_type, "ctx": label, "ctx_size": ctx_size,
                                "pp": best['pp_speed'], "tg": best['tg_speed']})
                print(f"  BEST: PP={best['pp_speed']:.2f} t/s | TG={best['tg_speed']:.2f} t/s")
            else:
                print(f"  {kv_type.upper()} FAILED inference at {label}")
                break

    # Summary
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'KV Type':<10} {'Context':<10} {'PP (t/s)':<12} {'TG (t/s)':<12}")
    print("-" * 44)
    for r in results:
        print(f"{r['kv_type']:<10} {r['ctx']:<10} {r['pp']:<12.2f} {r['tg']:<12.2f}")

    print(f"\n{'='*70}")
    print("MAX CONTEXT REACHED")
    print(f"{'='*70}")
    for kv in KV_TYPES:
        ctx = max_ctx.get(kv, 0)
        label = fmt_ctx(ctx) if ctx > 0 else "NONE"
        print(f"  {kv.upper():<10} {label}")

    # Memory estimates
    bpv = {"f16": 16.0, "q4_0": 4.5, "tq3_0": 3.5}
    # Gemma4 26B KV: 5 non-SWA layers × 512 + 25 SWA layers × 256 = 8960 dims per token pair (K+V)
    kv_dims_per_token = (5 * 512 + 25 * 256) * 2  # ×2 for K+V = 17920
    print(f"\n{'='*70}")
    print("ESTIMATED KV MEMORY")
    print(f"{'='*70}")
    for kv in KV_TYPES:
        ctx = max_ctx.get(kv, 0)
        if ctx > 0:
            mem_bytes = ctx * kv_dims_per_token * bpv[kv] / 8
            mem_gib = mem_bytes / (1024**3)
            print(f"  {kv.upper():<10} at {fmt_ctx(ctx):<6}: ~{mem_gib:.1f} GiB KV cache ({bpv[kv]} bpv)")

    print("\nDone!")

if __name__ == "__main__":
    main()
