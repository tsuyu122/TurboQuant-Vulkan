"""Benchmark runner: sequentially launches ``llama-server`` once per KV config,
runs the full 100-prompt suite against it, then shuts down.

OOM / VRAM fallback
-------------------
For each KV config we try to launch the server at ``c = 2048`` first, and
on launch failure retry at 1024, 512, 256, 128 in turn. Every prompt whose
``ctx_tier`` exceeds the launched ``c`` is skipped (``oom_skipped=True``)
so the output still contains a complete row for every (model, prompt) pair.

Usage
-----
    py -m bench_suite.runner --out bench_suite/results/raw.json

Options
-------
    --configs f16 tq3_0 tq3_0/tq2_0 ...   subset of KV configs to run
    --ngl N                                layer-offload count (default 24)
    --model PATH                           GGUF model (default gemma-4-26B Q4_K_M)
    --port N                               server port (default 8899)
    --limit N                              run only the first N prompts (smoke test)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Force UTF-8 on stdout/stderr so non-ASCII glyphs do not crash when the
# output is redirected to a file on Windows (default cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
except Exception:
    pass

from bench_suite.prompts import PROMPTS, CTX_TIERS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER_BIN = ROOT / "llama_src" / "build_vulkan" / "bin" / "Release" / "llama-server.exe"
DEFAULT_MODEL = ROOT / "models" / "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"

# KV configs: (label, ctk, ctv)
# KV configs: (label, ctk, ctv)
# Trimmed to the tq3_1 vs new-tq3_2 comparison only — the other configs are
# already characterised in the previous overnight run (raw.json.baseline).
KV_CONFIGS: dict[str, tuple[str, str]] = {
    "tq3_1":        ("tq3_0", "tq2_0"),    # published baseline (mixed KV)
    "tq3_2":        ("tq3_0", "tq3_2"),    # new MSE-optimized V scale
}

# Descending ladder — try the largest context the GPU will accept, then
# progressively halve until a launch succeeds. 1M and 2M tiers are aspirational:
# on a 12 GB card with a 26B Q4_K_M model they will almost certainly fall back.
FALLBACK_CTX_LADDER = (
    2_000_000, 1_000_000,
    524_288, 262_144, 131_072, 65_536, 32_768, 16_384, 8_192,
    4_096, 2_048, 1_024, 512, 256, 128,
)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 3.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""
    except Exception:
        return 0, b""


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ServerHandle:
    proc:        subprocess.Popen
    port:        int
    ctk:         str
    ctv:         str
    ctx:         int
    stderr_path: pathlib.Path


def _wait_ready(port: int, proc: subprocess.Popen, total_timeout: float = 180.0) -> bool:
    """Poll /health until it reports 200, or the server dies."""
    deadline = time.monotonic() + total_timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        code, _ = _http_get(url, timeout=2.0)
        if code == 200:
            return True
        time.sleep(1.0)
    return False


def start_server(
    model:       pathlib.Path,
    ctk:         str,
    ctv:         str,
    ctx:         int,
    port:        int,
    ngl:         int,
    log_dir:     pathlib.Path,
) -> ServerHandle | None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = log_dir / f"server_{ctk}_{ctv}_c{ctx}.log"
    cmd = [
        str(SERVER_BIN),
        "-m", str(model),
        "-ngl", str(ngl),
        "-fa", "1",
        "-ctk", ctk, "-ctv", ctv,
        "-c", str(ctx),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-np", "1",
        "--no-webui",
        "--log-disable",
    ]
    print(f"    launching: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    stderr_file = stderr_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
        cwd=str(ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    if not _wait_ready(port, proc):
        print(f"    server FAILED to start at ctx={ctx} (stderr: {stderr_path})", flush=True)
        _kill_server(proc)
        stderr_file.close()
        return None
    # Pre-flight: /health may return 200 before the KV cache is actually usable
    # (llama.cpp allocates lazily). Fire a tiny chat request and make sure it
    # returns something without timing out or crashing the server.
    if not _preflight(port, proc):
        print(f"    preflight FAILED at ctx={ctx} (server alive={proc.poll() is None}); falling back", flush=True)
        _kill_server(proc)
        stderr_file.close()
        return None
    print(f"    server ready on port {port} (ctx={ctx}, ctk={ctk}, ctv={ctv})", flush=True)
    return ServerHandle(proc=proc, port=port, ctk=ctk, ctv=ctv, ctx=ctx, stderr_path=stderr_path)


def _preflight(port: int, proc: subprocess.Popen, timeout_s: float = 60.0) -> bool:
    """Send a trivial chat completion to confirm the server can actually serve.
    Catches the case where /health returns 200 but the KV cache is unusable."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages":    [{"role": "user", "content": "ping"}],
        "max_tokens":  1,
        "temperature": 0.0,
        "stream":      False,
    }
    try:
        _http_post_json(url, payload, timeout=timeout_s)
    except Exception as exc:
        print(f"    preflight error: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        return False
    return proc.poll() is None


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def launch_with_fallback(
    model:  pathlib.Path,
    ctk:    str,
    ctv:    str,
    port:   int,
    ngl:    int,
    log_dir: pathlib.Path,
) -> ServerHandle | None:
    ladder = FALLBACK_CTX_LADDER
    for i, ctx in enumerate(ladder):
        handle = start_server(model, ctk, ctv, ctx, port, ngl, log_dir)
        if handle is not None:
            return handle
        nxt = ladder[i + 1] if i + 1 < len(ladder) else None
        print(f"    -> falling back to ctx={nxt}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def run_prompt(
    handle:    ServerHandle,
    spec:      dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    # Use the OpenAI-compatible endpoint so llama-server applies the model's
    # chat template (Gemma, Llama3, etc.) automatically. Sending raw text to
    # /completion produces degenerate output on chat-tuned models.
    url = f"http://127.0.0.1:{handle.port}/v1/chat/completions"
    payload = {
        "messages":    [{"role": "user", "content": spec["prompt"]}],
        "max_tokens":  spec["n_predict"],
        "temperature": 0.0,
        "top_k":       1,
        "stream":      False,
    }
    t0 = time.monotonic()
    try:
        resp = _http_post_json(url, payload, timeout=timeout_s)
    except Exception as exc:
        return {
            "id":            spec["id"],
            "category":      spec["category"],
            "ctx_tier":      spec["ctx_tier"],
            "prompt":        spec["prompt"],
            "response":      "",
            "error":         f"{type(exc).__name__}: {str(exc)[:200]}",
            "elapsed_s":     time.monotonic() - t0,
            "tokens_out":    0,
            "tokens_per_s":  0.0,
            "oom_skipped":   False,
        }
    elapsed = time.monotonic() - t0
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    finish_reason = choice.get("finish_reason", "") or ""
    usage = resp.get("usage") or {}
    tokens_out = int(usage.get("completion_tokens", 0) or 0)
    timings = resp.get("timings") or {}
    tps = float(timings.get("predicted_per_second", 0.0) or 0.0)
    if tps == 0.0 and elapsed > 0 and tokens_out > 0:
        tps = tokens_out / elapsed
    return {
        "id":            spec["id"],
        "category":      spec["category"],
        "ctx_tier":      spec["ctx_tier"],
        "prompt":        spec["prompt"],
        "response":      content,
        "reasoning":     reasoning,
        "finish_reason": finish_reason,
        "error":         "",
        "elapsed_s":     elapsed,
        "tokens_out":    tokens_out,
        "tokens_per_s":  tps,
        "oom_skipped":   False,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_one_model(
    label:    str,
    ctk:      str,
    ctv:      str,
    model:    pathlib.Path,
    ngl:      int,
    port:     int,
    log_dir:  pathlib.Path,
    prompts:  list[dict[str, Any]],
    timeout_s: float,
    resume_results: list[dict[str, Any]] | None = None,
    on_result: Any = None,  # callable() -> None, called after each prompt appended
) -> dict[str, Any]:
    # --- resume: figure out which prompts are already done ---
    # Only skip prompts that completed without error, or were legitimately
    # OOM-skipped. Prompts that errored (e.g. timeout) will be retried.
    done_ids: set[str] = set()
    if resume_results:
        done_ids = {
            r["id"] for r in resume_results
            if r.get("oom_skipped") or (not r.get("error"))
        }

    prompts_todo = [p for p in prompts if p["id"] not in done_ids]
    n_skipped_resume = len(prompts) - len(prompts_todo)

    print(f"\n[{label}] ctk={ctk} ctv={ctv}", flush=True)
    if n_skipped_resume:
        print(f"    RESUME: {n_skipped_resume}/{len(prompts)} prompts already done, "
              f"continuing from prompt {n_skipped_resume + 1}", flush=True)

    # Pre-populate results with already-completed entries in original order.
    # Exclude errored (non-oom) entries — they will be retried.
    results: list[dict[str, Any]] = [
        r for r in (resume_results or [])
        if r.get("oom_skipped") or (not r.get("error"))
    ]

    if not prompts_todo:
        print(f"    All {len(prompts)} prompts already complete — skipping server launch.", flush=True)
        loaded_ctx = (resume_results[0].get("loaded_ctx") if resume_results else None)
        return {
            "label":       label,
            "ctk":         ctk,
            "ctv":         ctv,
            "loaded_ctx":  loaded_ctx,
            "failed_launch": False,
            "results":     results,
        }

    handle = launch_with_fallback(model, ctk, ctv, port, ngl, log_dir)
    model_record: dict[str, Any] = {
        "label":       label,
        "ctk":         ctk,
        "ctv":         ctv,
        "loaded_ctx":  handle.ctx if handle else None,
        "failed_launch": handle is None,
        "results":     results,
    }
    if handle is None:
        print(f"    SKIPPING {label} — server would not launch at any ctx.", flush=True)
        for spec in prompts_todo:
            results.append({
                "id":            spec["id"],
                "category":      spec["category"],
                "ctx_tier":      spec["ctx_tier"],
                "prompt":        spec["prompt"],
                "response":      "",
                "error":         "server_launch_failed",
                "elapsed_s":     0.0,
                "tokens_out":    0,
                "tokens_per_s":  0.0,
                "oom_skipped":   True,
            })
        return model_record

    try:
        total = len(prompts)
        offset = n_skipped_resume  # display counter offset
        for i, spec in enumerate(prompts_todo, 1):
            display_i = offset + i
            if spec["ctx_tier"] > handle.ctx:
                results.append({
                    "id":            spec["id"],
                    "category":      spec["category"],
                    "ctx_tier":      spec["ctx_tier"],
                    "prompt":        spec["prompt"],
                    "response":      "",
                    "error":         f"ctx_tier {spec['ctx_tier']} > loaded {handle.ctx}",
                    "elapsed_s":     0.0,
                    "tokens_out":    0,
                    "tokens_per_s":  0.0,
                    "oom_skipped":   True,
                })
                print(f"  [{display_i:3d}/{total}] {spec['id']} SKIP (tier {spec['ctx_tier']} > ctx {handle.ctx})", flush=True)
                continue
            r = run_prompt(handle, spec, timeout_s)
            results.append(r)
            if on_result is not None:
                on_result(model_record)
            if r["error"]:
                print(f"  [{display_i:3d}/{total}] {spec['id']} ERR  {r['error'][:80]}", flush=True)
            else:
                print(f"  [{display_i:3d}/{total}] {spec['id']} ok   "
                      f"{r['tokens_out']:4d} tok  {r['tokens_per_s']:6.2f} t/s  {r['elapsed_s']:6.1f}s", flush=True)
    finally:
        _kill_server(handle.proc)
    return model_record


def main() -> int:
    ap = argparse.ArgumentParser(description="TurboQuant KV-cache benchmark suite.")
    ap.add_argument("--configs", nargs="+", default=list(KV_CONFIGS.keys()),
                    help=f"KV configs to run (default: all). Options: {list(KV_CONFIGS.keys())}")
    ap.add_argument("--ngl",     type=int,  default=24)
    ap.add_argument("--port",    type=int,  default=8899)
    ap.add_argument("--model",   type=str,  default=str(DEFAULT_MODEL))
    ap.add_argument("--out",     type=str,  default="bench_suite/results/raw.json")
    ap.add_argument("--limit",   type=int,  default=0, help="run only first N prompts (0 = all)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-prompt HTTP timeout seconds")
    args = ap.parse_args()

    model_path = pathlib.Path(args.model).resolve()
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1
    if not SERVER_BIN.exists():
        print(f"server binary not found: {SERVER_BIN}", file=sys.stderr)
        return 1

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "server_logs"

    prompts = PROMPTS if args.limit == 0 else PROMPTS[:args.limit]
    print(f"Running {len(prompts)} prompts across {len(args.configs)} KV configs", flush=True)
    print(f"Model:   {model_path}")
    print(f"Server:  {SERVER_BIN}")
    print(f"Out:     {out_path}")
    print(f"Logs:    {log_dir}")

    # --- Load partial checkpoint for resume ---
    checkpoint: dict[str, Any] = {}
    if out_path.exists():
        try:
            checkpoint = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"\nFound existing output — will resume from checkpoint: {out_path}", flush=True)
        except Exception as exc:
            print(f"Warning: could not load checkpoint ({exc}), starting fresh.", flush=True)
    # Build a lookup: label -> existing run record
    checkpoint_runs: dict[str, dict[str, Any]] = {
        r["label"]: r for r in checkpoint.get("runs", [])
    }

    run_records: list[dict[str, Any]] = []
    for label in args.configs:
        if label not in KV_CONFIGS:
            print(f"unknown config '{label}', skipping", file=sys.stderr)
            continue
        ctk, ctv = KV_CONFIGS[label]
        existing = checkpoint_runs.get(label)
        resume_results = existing["results"] if existing else None

        # Closure that persists the current snapshot after every prompt.
        def _on_result(in_progress_rec: dict[str, Any]) -> None:
            out_path.write_text(json.dumps({
                "model":      str(model_path.name),
                "ngl":        args.ngl,
                "prompts":    len(prompts),
                "kv_configs": args.configs,
                "runs":       run_records + [in_progress_rec],
            }, indent=2, ensure_ascii=False), encoding="utf-8")

        rec = run_one_model(label, ctk, ctv, model_path, args.ngl, args.port, log_dir, prompts, args.timeout,
                            resume_results=resume_results, on_result=_on_result)
        run_records.append(rec)
        # Persist after each model so a mid-run crash keeps partial data.
        out_path.write_text(json.dumps({
            "model":      str(model_path.name),
            "ngl":        args.ngl,
            "prompts":    len(prompts),
            "kv_configs": args.configs,
            "runs":       run_records,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Raw results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
