#!/usr/bin/env python3
# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
"""
TurboQuant-Vulkan — Fully Automated Resumable Benchmark System

Runs all quantization x context-size configurations, collects metrics,
and saves results for later evaluation. Designed to survive power failures.

Usage:
    py bench/run_benchmarks.py                 # Full run
    py bench/run_benchmarks.py --resume        # Resume interrupted run
    py bench/run_benchmarks.py --dry-run       # Preview configurations
    py bench/run_benchmarks.py --only tq3_0    # Test single quant type
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
LLAMA_SERVER = BASE_DIR / "llama_src" / "build_vulkan" / "bin" / "Release" / "llama-server.exe"
MODEL = BASE_DIR / "models" / "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"

BENCH_DIR = BASE_DIR / "bench"
RUNS_DIR = BENCH_DIR / "runs"
RESULTS_DIR = BENCH_DIR / "results"
REPORTS_DIR = BENCH_DIR / "reports"
QUALITY_DIR = BENCH_DIR / "quality"
PROMPTS_FILE = BENCH_DIR / "prompts" / "prompts.json"

PORT = 8090
NGL = 30
MAX_RETRIES = 2
MAX_RUNS_PER_CONFIG = 10
GENERATION_TOKENS = 512
SERVER_STARTUP_TIMEOUT = 120  # seconds
QUERY_TIMEOUT = 300  # seconds

# Quantization configurations: (name, ctk, ctv, description)
QUANT_CONFIGS = [
    ("f16",      "f16",   "f16",   "FP16 baseline — full precision KV cache"),
    ("q8_0",     "q8_0",  "q8_0",  "Q8_0 — 8-bit symmetric quantization"),
    ("q4_0",     "q4_0",  "q4_0",  "Q4_0 — 4-bit symmetric quantization"),
    ("tq3_0",    "tq3_0", "tq3_0", "TQ3_0 — TurboQuant 3-bit (Lloyd-Max codebook)"),
    ("tq2_0",    "tq2_0", "tq2_0", "TQ2_0 — TurboQuant 2-bit (Lloyd-Max codebook)"),
    ("tq3_1",    "tq3_0", "tq2_0", "TQ3_1 — TurboQuant Mixed (K=TQ3_0, V=TQ2_0)"),
]

# Context sizes to test (ascending order for early-stop logic)
CONTEXT_SIZES = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


# ─── Utilities ────────────────────────────────────────────────────────────────

def atomic_write_json(path: Path, data: dict):
    """Write JSON atomically via temp file + rename to survive power failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    shutil.move(str(tmp), str(path))


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_prompts() -> list:
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def config_id(quant_name: str, ctx: int) -> str:
    return f"{quant_name}_ctx{ctx}"


def run_id(quant_name: str, ctx: int, run_num: int) -> str:
    return f"{quant_name}_ctx{ctx}_run{run_num:02d}"


def output_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Server Management ─────────────────────────────────────────────────────

def start_server(ctk: str, ctv: str, ctx: int, stderr_path: Path) -> subprocess.Popen:
    """Start llama-server and wait for it to be ready."""
    args = [
        str(LLAMA_SERVER),
        "-m", str(MODEL),
        "-ngl", str(NGL),
        "-ctk", ctk,
        "-ctv", ctv,
        "-c", str(ctx),
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "-fa",
    ]
    stderr_file = open(stderr_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    return proc


def wait_for_server(timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll /health until server is ready."""
    for _ in range(timeout):
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def kill_server(proc: subprocess.Popen):
    """Kill the server process."""
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception:
        pass
    # Also kill any leftover instances
    if sys.platform == "win32":
        os.system("taskkill /F /IM llama-server.exe >nul 2>&1")
    time.sleep(2)


def kill_all_servers():
    """Kill all llama-server instances."""
    if sys.platform == "win32":
        os.system("taskkill /F /IM llama-server.exe >nul 2>&1")
    else:
        os.system("pkill -f llama-server 2>/dev/null")
    time.sleep(2)


# ─── Query ──────────────────────────────────────────────────────────────────

def query_server(prompt: str, max_tokens: int = GENERATION_TOKENS) -> dict:
    """Send a completion request and collect timing metrics."""
    body = json.dumps({
        "model": "gemma",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=QUERY_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "elapsed": time.perf_counter() - t0}

    elapsed = time.perf_counter() - t0
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    timings = data.get("timings", {})

    prompt_tokens = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0)
    gen_tokens = timings.get("predicted_n", 0)
    gen_ms = timings.get("predicted_ms", 0)

    result = {
        "content": content.strip(),
        "reasoning": reasoning.strip(),
        "prompt_tokens": prompt_tokens,
        "prompt_ms": prompt_ms,
        "prompt_tps": round(prompt_tokens / (prompt_ms / 1000), 2) if prompt_ms > 0 else 0,
        "gen_tokens": gen_tokens,
        "gen_ms": gen_ms,
        "gen_tps": round(gen_tokens / (gen_ms / 1000), 2) if gen_ms > 0 else 0,
        "ttft_ms": timings.get("prompt_ms", 0),
        "total_latency_ms": round(elapsed * 1000, 1),
        "elapsed_sec": round(elapsed, 2),
    }
    return result


# ─── VRAM Collection ────────────────────────────────────────────────────────

def get_vram_usage() -> dict:
    """Get GPU VRAM usage from the server's /slots endpoint or system tools."""
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
        # Health endpoint doesn't give VRAM, but we can try /props
        resp2 = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5)
        props = json.loads(resp2.read())
        return {"vram_info": "collected_from_props", "props": props}
    except Exception:
        pass
    return {"vram_info": "unavailable"}


# ─── Run Management ─────────────────────────────────────────────────────────

def get_run_dir(rid: str) -> Path:
    return RUNS_DIR / rid


def get_run_status(rid: str) -> str:
    status_path = get_run_dir(rid) / "status.json"
    if status_path.exists():
        data = load_json(status_path)
        return data.get("status", "unknown")
    return "not_exists"


def count_config_runs(quant_name: str, ctx: int) -> int:
    """Count completed + running runs for a configuration."""
    count = 0
    for i in range(1, MAX_RUNS_PER_CONFIG + 1):
        rid = run_id(quant_name, ctx, i)
        status = get_run_status(rid)
        if status in ("done", "running"):
            count += 1
    return count


def count_completed_runs(quant_name: str, ctx: int) -> int:
    """Count only completed runs."""
    count = 0
    for i in range(1, MAX_RUNS_PER_CONFIG + 1):
        rid = run_id(quant_name, ctx, i)
        if get_run_status(rid) == "done":
            count += 1
    return count


def is_config_failed(quant_name: str, ctx: int) -> bool:
    """Check if a config has failed all retries."""
    fail_count = 0
    for i in range(1, MAX_RUNS_PER_CONFIG + 1):
        rid = run_id(quant_name, ctx, i)
        if get_run_status(rid) == "failed":
            fail_count += 1
    return fail_count >= MAX_RETRIES + 1


def get_next_run_number(quant_name: str, ctx: int) -> int:
    """Find the next available run number."""
    for i in range(1, MAX_RUNS_PER_CONFIG + 1):
        rid = run_id(quant_name, ctx, i)
        status = get_run_status(rid)
        if status in ("not_exists", "running"):
            return i
    return -1


def should_skip_higher_contexts(quant_name: str, ctx: int, ctx_list: list) -> bool:
    """Check if we should skip this and higher contexts due to prior failure."""
    for prev_ctx in ctx_list:
        if prev_ctx >= ctx:
            break
        if is_config_failed(quant_name, prev_ctx):
            return True
    return False


# ─── Single Run Execution ──────────────────────────────────────────────────

def execute_run(quant_name: str, ctk: str, ctv: str, ctx: int,
                run_num: int, prompt_data: dict) -> dict:
    """Execute a single benchmark run. Returns result dict."""
    rid = run_id(quant_name, ctx, run_num)
    rdir = get_run_dir(rid)
    rdir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_data = {
        "quant_name": quant_name,
        "ctk": ctk,
        "ctv": ctv,
        "context_size": ctx,
        "run_number": run_num,
        "run_id": rid,
        "prompt_id": prompt_data["id"],
        "prompt_category": prompt_data["category"],
        "ngl": NGL,
        "model": str(MODEL.name),
        "max_tokens": GENERATION_TOKENS,
        "timestamp_start": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(rdir / "config.json", config_data)

    # Mark as running
    atomic_write_json(rdir / "status.json", {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    stderr_path = rdir / "stderr.log"
    proc = None

    try:
        # Start server
        log(f"  Starting server: {quant_name} ctx={ctx} ...")
        proc = start_server(ctk, ctv, ctx, stderr_path)

        if not wait_for_server():
            raise RuntimeError("Server failed to start within timeout")

        # Warmup
        log(f"  Warmup query...")
        warmup = query_server("Hello, respond with just 'Hi'.", max_tokens=10)
        if "error" in warmup:
            raise RuntimeError(f"Warmup failed: {warmup['error']}")
        time.sleep(1)

        # VRAM before
        vram_before = get_vram_usage()

        # Main query
        log(f"  Running prompt: {prompt_data['id']} ({prompt_data['category']})...")
        result = query_server(prompt_data["prompt"], max_tokens=GENERATION_TOKENS)

        if "error" in result:
            raise RuntimeError(f"Query failed: {result['error']}")

        # VRAM after
        vram_after = get_vram_usage()

        # Save output
        output_text = result.get("content", "")
        with open(rdir / "output.txt", "w", encoding="utf-8") as f:
            f.write(output_text)

        # Save stdout log (metrics)
        with open(rdir / "stdout.log", "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2))

        # Build metrics
        metrics = {
            "run_id": rid,
            "quant_name": quant_name,
            "context_size": ctx,
            "prompt_id": prompt_data["id"],
            "prompt_category": prompt_data["category"],
            "prompt_tps": result.get("prompt_tps", 0),
            "gen_tps": result.get("gen_tps", 0),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "gen_tokens": result.get("gen_tokens", 0),
            "ttft_ms": result.get("ttft_ms", 0),
            "total_latency_ms": result.get("total_latency_ms", 0),
            "output_hash": output_hash(output_text),
            "output_length_chars": len(output_text),
            "vram_before": vram_before,
            "vram_after": vram_after,
            "success": True,
        }
        atomic_write_json(rdir / "metrics.json", metrics)

        # Mark as done
        atomic_write_json(rdir / "status.json", {
            "status": "done",
            "started_at": config_data["timestamp_start"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

        log(f"  DONE: prompt={result.get('prompt_tps', 0):.1f} t/s, "
            f"gen={result.get('gen_tps', 0):.1f} t/s, "
            f"tokens={result.get('gen_tokens', 0)}")

        return metrics

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log(f"  FAILED: {error_msg}")

        # Check stderr for OOM
        oom = False
        if stderr_path.exists():
            try:
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
                if "out of memory" in stderr_text.lower() or "oom" in stderr_text.lower() or \
                   "vk_error_out_of_device_memory" in stderr_text.lower():
                    oom = True
            except Exception:
                pass

        atomic_write_json(rdir / "status.json", {
            "status": "failed",
            "error": error_msg,
            "oom": oom,
            "started_at": config_data.get("timestamp_start", ""),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        })

        atomic_write_json(rdir / "metrics.json", {
            "run_id": rid,
            "quant_name": quant_name,
            "context_size": ctx,
            "prompt_id": prompt_data["id"],
            "success": False,
            "error": error_msg,
            "oom": oom,
        })

        return {"success": False, "error": error_msg, "oom": oom}

    finally:
        if proc:
            kill_server(proc)


# ─── Build Run Plan ─────────────────────────────────────────────────────────

def build_run_plan(prompts: list, only_quant: str = None,
                   min_runs: int = 3) -> list:
    """Build list of (quant_name, ctk, ctv, ctx, run_num, prompt) tuples."""
    plan = []

    for quant_name, ctk, ctv, desc in QUANT_CONFIGS:
        if only_quant and quant_name != only_quant:
            continue

        for ctx in CONTEXT_SIZES:
            cid = config_id(quant_name, ctx)

            # Check if we should skip due to prior lower-ctx failure
            if should_skip_higher_contexts(quant_name, ctx, CONTEXT_SIZES):
                log(f"  SKIP {cid}: lower context failed for {quant_name}")
                continue

            # Count existing completed runs
            completed = count_completed_runs(quant_name, ctx)
            if completed >= MAX_RUNS_PER_CONFIG:
                continue

            # Check if config is permanently failed
            if is_config_failed(quant_name, ctx):
                continue

            # Schedule remaining runs
            runs_needed = min(min_runs, MAX_RUNS_PER_CONFIG) - completed
            for _ in range(max(runs_needed, 0)):
                run_num = get_next_run_number(quant_name, ctx)
                if run_num < 0:
                    break

                # Pick a random prompt
                prompt = random.choice(prompts)
                plan.append((quant_name, ctk, ctv, ctx, run_num, prompt))

    return plan


# ─── Main Benchmark Loop ──────────────────────────────────────────────────

def run_benchmarks(args):
    """Main benchmark execution."""
    # Ensure directories exist
    for d in [RUNS_DIR, RESULTS_DIR, REPORTS_DIR, QUALITY_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts()
    log(f"Loaded {len(prompts)} prompts")
    log(f"Model: {MODEL.name}")
    log(f"Server: {LLAMA_SERVER}")
    log(f"GPU layers: {NGL}")
    log(f"Max runs per config: {MAX_RUNS_PER_CONFIG}")
    log(f"Generation tokens: {GENERATION_TOKENS}")

    if not LLAMA_SERVER.exists():
        log(f"ERROR: llama-server not found at {LLAMA_SERVER}")
        sys.exit(1)
    if not MODEL.exists():
        log(f"ERROR: Model not found at {MODEL}")
        sys.exit(1)

    # Kill any leftover servers
    kill_all_servers()

    # Track failed configs to implement early-stop
    failed_configs = set()
    # Track scheduled run numbers during dry-run
    scheduled_runs = {}  # (quant_name, ctx) -> set of run_nums

    total_runs = 0
    total_success = 0
    total_failed = 0
    skipped_higher = set()  # quant names where we skip higher ctx

    log("")
    log("=" * 80)
    log("TurboQuant-Vulkan — Automated Benchmark System")
    log("=" * 80)
    log("")

    for quant_name, ctk, ctv, desc in QUANT_CONFIGS:
        if args.only and quant_name != args.only:
            continue

        log(f"\n{'='*80}")
        log(f"Quantization: {quant_name} ({desc})")
        log(f"{'='*80}")

        for ctx in CONTEXT_SIZES:
            cid = config_id(quant_name, ctx)

            # Early-stop: skip if lower context failed for this quant
            if quant_name in skipped_higher:
                log(f"\n  SKIP {cid}: lower context failed, skipping all higher")
                continue

            # Check existing state
            completed = count_completed_runs(quant_name, ctx)
            if completed >= args.runs_per_config:
                log(f"\n  SKIP {cid}: already have {completed} completed runs")
                continue

            if is_config_failed(quant_name, ctx):
                log(f"\n  SKIP {cid}: permanently failed")
                skipped_higher.add(quant_name)
                continue

            log(f"\n  Config: {cid} (completed: {completed}/{args.runs_per_config})")

            runs_needed = args.runs_per_config - completed
            consecutive_failures = 0

            for run_i in range(runs_needed):
                # Find next available run number (accounting for scheduled but not yet executed)
                sched_key = (quant_name, ctx)
                if sched_key not in scheduled_runs:
                    scheduled_runs[sched_key] = set()

                run_num = -1
                for candidate in range(1, MAX_RUNS_PER_CONFIG + 1):
                    rid_candidate = run_id(quant_name, ctx, candidate)
                    existing_status = get_run_status(rid_candidate)
                    if existing_status == "done":
                        continue
                    if candidate in scheduled_runs[sched_key]:
                        continue
                    run_num = candidate
                    break

                if run_num < 0 or run_num > MAX_RUNS_PER_CONFIG:
                    break

                scheduled_runs[sched_key].add(run_num)
                rid = run_id(quant_name, ctx, run_num)

                # Pick random prompt
                prompt = random.choice(prompts)

                log(f"\n  --- Run {run_num} ({rid}) ---")
                log(f"      Prompt: {prompt['id']} ({prompt['category']})")

                if args.dry_run:
                    log(f"      [DRY RUN] Would execute")
                    continue

                total_runs += 1
                result = execute_run(quant_name, ctk, ctv, ctx, run_num, prompt)

                if result.get("success"):
                    total_success += 1
                    consecutive_failures = 0
                else:
                    total_failed += 1
                    consecutive_failures += 1

                    if result.get("oom"):
                        log(f"      OOM detected — stopping higher contexts for {quant_name}")
                        skipped_higher.add(quant_name)
                        break

                    if consecutive_failures > MAX_RETRIES:
                        log(f"      {MAX_RETRIES + 1} consecutive failures — marking config as failed")
                        skipped_higher.add(quant_name)
                        break

            if quant_name in skipped_higher:
                break  # Skip remaining context sizes

    log(f"\n{'='*80}")
    log(f"Benchmark Complete")
    log(f"  Total runs: {total_runs}")
    log(f"  Successful: {total_success}")
    log(f"  Failed:     {total_failed}")
    log(f"{'='*80}")

    # Generate results
    if not args.dry_run:
        generate_results()
        generate_quality_dataset(prompts)
        generate_report()


# ─── Results Aggregation ──────────────────────────────────────────────────

def collect_all_metrics() -> list:
    """Collect metrics from all completed runs."""
    metrics = []
    if not RUNS_DIR.exists():
        return metrics
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        status_path = run_dir / "status.json"
        if metrics_path.exists() and status_path.exists():
            status = load_json(status_path)
            if status.get("status") == "done":
                m = load_json(metrics_path)
                if m.get("success"):
                    metrics.append(m)
    return metrics


def generate_results():
    """Generate CSV result files from completed runs."""
    log("\nGenerating result files...")
    metrics = collect_all_metrics()
    if not metrics:
        log("  No completed runs found")
        return

    # throughput.csv
    throughput_path = RESULTS_DIR / "throughput.csv"
    with open(throughput_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "quant_name", "context_size", "prompt_id",
            "prompt_tps", "gen_tps", "prompt_tokens", "gen_tokens",
            "ttft_ms", "total_latency_ms"
        ])
        for m in metrics:
            writer.writerow([
                m.get("run_id"), m.get("quant_name"), m.get("context_size"),
                m.get("prompt_id"), m.get("prompt_tps"), m.get("gen_tps"),
                m.get("prompt_tokens"), m.get("gen_tokens"),
                m.get("ttft_ms"), m.get("total_latency_ms"),
            ])
    log(f"  Saved: {throughput_path}")

    # stability.csv
    all_runs = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        status_path = run_dir / "status.json"
        config_path = run_dir / "config.json"
        if status_path.exists() and config_path.exists():
            status = load_json(status_path)
            config = load_json(config_path)
            all_runs.append({
                "run_id": run_dir.name,
                "quant_name": config.get("quant_name"),
                "context_size": config.get("context_size"),
                "status": status.get("status"),
                "error": status.get("error", ""),
                "oom": status.get("oom", False),
            })

    stability_path = RESULTS_DIR / "stability.csv"
    with open(stability_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "quant_name", "context_size", "status", "error", "oom"])
        for r in all_runs:
            writer.writerow([
                r["run_id"], r["quant_name"], r["context_size"],
                r["status"], r["error"], r["oom"],
            ])
    log(f"  Saved: {stability_path}")

    # quality_index.csv — per-config aggregated
    from collections import defaultdict
    config_metrics = defaultdict(list)
    for m in metrics:
        key = (m["quant_name"], m["context_size"])
        config_metrics[key].append(m)

    quality_path = RESULTS_DIR / "quality_index.csv"
    with open(quality_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "quant_name", "context_size", "num_runs",
            "avg_prompt_tps", "avg_gen_tps", "avg_ttft_ms",
            "min_gen_tps", "max_gen_tps", "avg_output_length",
        ])
        for (qn, ctx), mlist in sorted(config_metrics.items()):
            n = len(mlist)
            avg_ptps = sum(m["prompt_tps"] for m in mlist) / n
            avg_gtps = sum(m["gen_tps"] for m in mlist) / n
            avg_ttft = sum(m.get("ttft_ms", 0) for m in mlist) / n
            min_gtps = min(m["gen_tps"] for m in mlist)
            max_gtps = max(m["gen_tps"] for m in mlist)
            avg_outlen = sum(m.get("output_length_chars", 0) for m in mlist) / n
            writer.writerow([
                qn, ctx, n,
                f"{avg_ptps:.2f}", f"{avg_gtps:.2f}", f"{avg_ttft:.1f}",
                f"{min_gtps:.2f}", f"{max_gtps:.2f}", f"{avg_outlen:.0f}",
            ])
    log(f"  Saved: {quality_path}")


def generate_quality_dataset(prompts: list):
    """Generate dataset for external LLM evaluation."""
    log("\nGenerating quality evaluation dataset...")
    dataset = []

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        status = load_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
        if status.get("status") != "done":
            continue

        config = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
        output_path = run_dir / "output.txt"
        output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""

        # Find original prompt
        prompt_id = config.get("prompt_id", "")
        prompt_text = ""
        for p in prompts:
            if p["id"] == prompt_id:
                prompt_text = p["prompt"]
                break

        dataset.append({
            "run_id": run_dir.name,
            "quant_name": config.get("quant_name"),
            "context_size": config.get("context_size"),
            "prompt_id": prompt_id,
            "prompt_category": config.get("prompt_category"),
            "prompt": prompt_text,
            "output": output,
        })

    dataset_path = QUALITY_DIR / "dataset_for_eval.json"
    atomic_write_json(dataset_path, dataset)
    log(f"  Saved: {dataset_path} ({len(dataset)} entries)")


def generate_report():
    """Generate the final markdown report."""
    log("\nGenerating final report...")
    metrics = collect_all_metrics()
    if not metrics:
        log("  No data for report")
        return

    from collections import defaultdict
    config_metrics = defaultdict(list)
    for m in metrics:
        key = (m["quant_name"], m["context_size"])
        config_metrics[key].append(m)

    # Collect all runs for stability
    all_statuses = defaultdict(lambda: {"done": 0, "failed": 0, "oom": 0})
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        status = load_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
        config = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
        qn = config.get("quant_name", "unknown")
        ctx = config.get("context_size", 0)
        key = (qn, ctx)
        st = status.get("status", "unknown")
        if st == "done":
            all_statuses[key]["done"] += 1
        elif st == "failed":
            all_statuses[key]["failed"] += 1
            if status.get("oom"):
                all_statuses[key]["oom"] += 1

    # Find max context per quant
    max_ctx_per_quant = {}
    for (qn, ctx), mlist in config_metrics.items():
        if qn not in max_ctx_per_quant or ctx > max_ctx_per_quant[qn]:
            max_ctx_per_quant[qn] = ctx

    lines = []
    lines.append("# TurboQuant-Vulkan — Benchmark Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Model: {MODEL.name}  ")
    lines.append(f"GPU Layers: {NGL}  ")
    lines.append(f"Generation Tokens: {GENERATION_TOKENS}  ")
    lines.append(f"Max Runs Per Config: {MAX_RUNS_PER_CONFIG}\n")

    # Performance table
    lines.append("\n## Performance Summary\n")
    lines.append("| Quant | Context | Runs | Avg Prompt t/s | Avg Gen t/s | Avg TTFT (ms) | Avg Latency (ms) |")
    lines.append("|-------|---------|------|----------------|-------------|---------------|-------------------|")

    for (qn, ctx), mlist in sorted(config_metrics.items()):
        n = len(mlist)
        avg_ptps = sum(m["prompt_tps"] for m in mlist) / n
        avg_gtps = sum(m["gen_tps"] for m in mlist) / n
        avg_ttft = sum(m.get("ttft_ms", 0) for m in mlist) / n
        avg_lat = sum(m.get("total_latency_ms", 0) for m in mlist) / n
        ctx_str = f"{ctx//1024}K" if ctx >= 1024 else str(ctx)
        lines.append(f"| {qn} | {ctx_str} | {n} | {avg_ptps:.1f} | {avg_gtps:.1f} | {avg_ttft:.0f} | {avg_lat:.0f} |")

    # Max context
    lines.append("\n## Maximum Stable Context\n")
    lines.append("| Quant | Max Context | Status |")
    lines.append("|-------|-------------|--------|")
    for qn, ctx in sorted(max_ctx_per_quant.items()):
        ctx_str = f"{ctx//1024}K" if ctx >= 1024 else str(ctx)
        lines.append(f"| {qn} | {ctx_str} | Stable |")

    # Stability
    lines.append("\n## Stability Summary\n")
    lines.append("| Quant | Context | Success | Failed | OOM |")
    lines.append("|-------|---------|---------|--------|-----|")
    for (qn, ctx), counts in sorted(all_statuses.items()):
        ctx_str = f"{ctx//1024}K" if ctx >= 1024 else str(ctx)
        lines.append(f"| {qn} | {ctx_str} | {counts['done']} | {counts['failed']} | {counts['oom']} |")

    # Failure notes
    lines.append("\n## Failure Notes\n")
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        status = load_json(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
        if status.get("status") == "failed":
            error = status.get("error", "unknown")
            oom = " (OOM)" if status.get("oom") else ""
            lines.append(f"- `{run_dir.name}`: {error}{oom}")

    report_path = REPORTS_DIR / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"  Saved: {report_path}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def _update_config(ngl_val, port_val, gen_tokens_val):
    global NGL, PORT, GENERATION_TOKENS
    NGL = ngl_val
    PORT = port_val
    GENERATION_TOKENS = gen_tokens_val


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant-Vulkan Automated Benchmark System"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous interrupted run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview run plan without executing")
    parser.add_argument("--only", type=str, default=None,
                        help="Only test specific quant type (e.g., tq3_0)")
    parser.add_argument("--runs-per-config", type=int, default=3,
                        help="Number of runs per configuration (max 10)")
    parser.add_argument("--ngl", type=int, default=NGL,
                        help="Number of GPU layers")
    parser.add_argument("--port", type=int, default=PORT,
                        help="Server port")
    parser.add_argument("--gen-tokens", type=int, default=GENERATION_TOKENS,
                        help="Tokens to generate per run")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate reports from existing data")
    args = parser.parse_args()

    # Update module-level config from CLI args
    _update_config(args.ngl, args.port, args.gen_tokens)

    if args.runs_per_config > MAX_RUNS_PER_CONFIG:
        log(f"WARNING: Capping runs_per_config to {MAX_RUNS_PER_CONFIG}")
        args.runs_per_config = MAX_RUNS_PER_CONFIG

    if args.report_only:
        prompts = load_prompts()
        generate_results()
        generate_quality_dataset(prompts)
        generate_report()
        return

    run_benchmarks(args)


if __name__ == "__main__":
    main()
