# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
# See LICENSE file for details.
"""
TurboQuant Vulkan — Interactive Launcher
Select model, KV compression type, context size, GPU layers and more.
Builds the llama-server command and runs it for you.

Usage:
    python scripts/launcher.py
"""

import os
import sys
import glob
import subprocess
import platform

IS_WINDOWS = platform.system() == "Windows"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
BUILD_DIR = os.path.join(REPO_ROOT, "llama.cpp", "build")

KV_TYPES = {
    "1": ("f16",   "F16  — full precision (baseline, max VRAM)"),
    "2": ("tq3_0", "TQ3_0 — 3-bit, 8 centroids (93% quality, -78% VRAM) ← recommended"),
    "3": ("tq2_0", "TQ2_0 — 2-bit, 4 centroids (50% quality, -84% VRAM, chat/summary only)"),
    "4": ("q8_0",  "Q8_0 — standard 8-bit (high quality, moderate VRAM)"),
    "5": ("q4_0",  "Q4_0 — standard 4-bit (good quality)"),
}

CONTEXT_SIZES = {
    "1": 2048,
    "2": 4096,
    "3": 8192,
    "4": 16384,
    "5": 32768,
    "6": 65536,
    "7": 131072,
    "8": "custom",
}


def sep():
    print("-" * 60)


def header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def find_server():
    """Locate the llama-server binary in the build directory."""
    exe_name = "llama-server.exe" if IS_WINDOWS else "llama-server"
    candidates = [
        os.path.join(BUILD_DIR, "bin", "Release", exe_name),
        os.path.join(BUILD_DIR, "bin", exe_name),
        os.path.join(BUILD_DIR, exe_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Search tree as last resort
    for root, _dirs, files in os.walk(BUILD_DIR):
        if exe_name in files:
            return os.path.join(root, exe_name)

    return None


def find_models():
    """Return list of .gguf files from the models/ directory."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(glob.glob(os.path.join(MODELS_DIR, "**", "*.gguf"), recursive=True))


def prompt(question, default=None):
    """Read a line from stdin, returning default on empty input."""
    if default is not None:
        question = f"{question} [{default}]: "
    else:
        question = f"{question}: "
    try:
        answer = input(question).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(0)
    return answer if answer else (str(default) if default is not None else "")


def select_model():
    models = find_models()
    if not models:
        print(f"\n[!] No .gguf files found in: {MODELS_DIR}")
        print("    Place your model files there and re-run.")
        print("    You can specify a path manually below.")
        path = prompt("Model path")
        if not path or not os.path.isfile(path):
            print("[ERROR] File not found.")
            sys.exit(1)
        return path

    print("\nAvailable models:")
    sep()
    for i, m in enumerate(models, 1):
        rel = os.path.relpath(m, REPO_ROOT)
        size_mb = os.path.getsize(m) / (1024 ** 2)
        print(f"  {i}. {rel}  ({size_mb:.0f} MB)")
    sep()

    while True:
        choice = prompt("Select model number")
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("    Invalid choice, try again.")


def select_kv_type():
    print("\nKV cache compression type:")
    sep()
    for key in sorted(KV_TYPES):
        flag, desc = KV_TYPES[key]
        print(f"  {key}. {desc}")
    sep()

    while True:
        choice = prompt("Select KV type", default="2")
        if choice in KV_TYPES:
            flag, desc = KV_TYPES[choice]
            return flag
        print("    Invalid choice, try again.")


def select_context():
    print("\nContext size (number of tokens):")
    sep()
    for key in sorted(CONTEXT_SIZES, key=int):
        val = CONTEXT_SIZES[key]
        if val == "custom":
            print(f"  {key}. Custom")
        else:
            print(f"  {key}. {val:,}")
    sep()

    while True:
        choice = prompt("Select context size", default="4")
        if choice in CONTEXT_SIZES:
            val = CONTEXT_SIZES[choice]
            if val == "custom":
                raw = prompt("Enter context size (tokens)")
                if raw.isdigit() and int(raw) > 0:
                    return int(raw)
                print("    Invalid value, try again.")
            else:
                return val
        print("    Invalid choice, try again.")


def select_gpu_layers():
    print("\nGPU layers (number of model layers to offload to GPU):")
    sep()
    print("  0. CPU only (0 layers)")
    print("  1. Partial offload (15 layers)")
    print("  2. Full offload (99 — all layers)")
    print("  3. Custom")
    sep()

    while True:
        choice = prompt("Select GPU layers", default="2")
        if choice == "0":
            return 0
        elif choice == "1":
            return 15
        elif choice == "2":
            return 99
        elif choice == "3":
            raw = prompt("Enter number of GPU layers")
            if raw.isdigit() and int(raw) >= 0:
                return int(raw)
            print("    Invalid value, try again.")
        else:
            print("    Invalid choice, try again.")


def select_port():
    raw = prompt("Server port", default=8080)
    try:
        port = int(raw)
        if 1 <= port <= 65535:
            return port
    except ValueError:
        pass
    print("    Invalid port — using 8080.")
    return 8080


def select_host():
    raw = prompt("Bind address (127.0.0.1 = local only, 0.0.0.0 = all interfaces)", default="127.0.0.1")
    return raw if raw else "127.0.0.1"


def main():
    header("TurboQuant Vulkan — Interactive Launcher")

    # --- Find llama-server ---
    server = find_server()
    if not server:
        print(f"\n[ERROR] llama-server not found in: {BUILD_DIR}")
        print("  Run 'python scripts/setup.py' first to build llama.cpp.")
        sys.exit(1)
    print(f"\nUsing binary: {os.path.relpath(server, REPO_ROOT)}")

    # --- Model ---
    header("Step 1: Select model")
    model = select_model()

    # --- KV type ---
    header("Step 2: KV cache compression")
    kv_type = select_kv_type()

    # --- Context ---
    header("Step 3: Context size")
    ctx = select_context()

    # --- GPU layers ---
    header("Step 4: GPU offload")
    ngl = select_gpu_layers()

    # --- Network ---
    header("Step 5: Server settings")
    port = select_port()
    host = select_host()

    # --- Flash Attention ---
    fa_choice = prompt("Enable Flash Attention? (y/n)", default="y").lower()
    use_fa = fa_choice not in ("n", "no")

    # --- Build command ---
    cmd = [
        server,
        "-m", model,
        "-c", str(ctx),
        "-ngl", str(ngl),
        "--host", host,
        "--port", str(port),
        "-ctk", kv_type,
        "-ctv", kv_type,
    ]
    if use_fa:
        cmd.append("-fa")

    header("Final command")
    print()
    print("  " + " \\\n    ".join(cmd))
    print()

    confirm = prompt("Run this command? (y/n)", default="y").lower()
    if confirm in ("n", "no"):
        print("\nAborted. Copy the command above to run manually.")
        sys.exit(0)

    print(f"\nStarting llama-server on http://{host}:{port}/\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
