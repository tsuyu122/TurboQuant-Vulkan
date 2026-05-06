# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
# See LICENSE file for details.
"""
TurboQuant Vulkan — Automated Setup
Downloads llama.cpp at the correct commit, applies tq3_0.patch,
builds with Vulkan, and creates the models/ directory.

Usage:
    python scripts/setup.py
"""

import os
import sys
import subprocess
import shutil
import platform

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_COMMIT = "073bb2c"
LLAMA_CPP_DIR = os.path.join(REPO_ROOT, "llama.cpp")
PATCH_FILE = os.path.join(REPO_ROOT, "tq3_0.patch")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

IS_WINDOWS = platform.system() == "Windows"


def step(msg):
    print(f"\n[=] {msg}")


def run(cmd, cwd=None, description=None):
    if description:
        print(f"    > {description}")
    print(f"    $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def check_dependency(name):
    if shutil.which(name) is None:
        print(f"[ERROR] '{name}' not found in PATH. Please install it first.")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  TurboQuant Vulkan — Automated Setup")
    print("=" * 60)

    # --- Check dependencies ---
    step("Checking dependencies")
    check_dependency("git")
    check_dependency("cmake")
    print("    git: OK")
    print("    cmake: OK")

    # --- Clone llama.cpp ---
    step("Clone llama.cpp")
    if os.path.isdir(LLAMA_CPP_DIR):
        print(f"    llama.cpp already exists at: {LLAMA_CPP_DIR}")
        print("    Skipping clone — delete the folder to re-clone.")
    else:
        run(
            ["git", "clone", "--depth", "1", LLAMA_CPP_URL, LLAMA_CPP_DIR],
            description="Cloning (shallow)...",
        )
        # Shallow clone may not have the exact commit; fetch it explicitly
        run(
            ["git", "fetch", "--unshallow"],
            cwd=LLAMA_CPP_DIR,
            description="Unshallowing to fetch full history for checkout...",
        )

    # --- Checkout pinned commit ---
    step(f"Checkout commit {LLAMA_CPP_COMMIT}")
    run(
        ["git", "checkout", LLAMA_CPP_COMMIT],
        cwd=LLAMA_CPP_DIR,
        description="Switching to tested commit...",
    )

    # --- Apply patch ---
    step("Apply tq3_0.patch")
    if not os.path.isfile(PATCH_FILE):
        print(f"[ERROR] Patch file not found: {PATCH_FILE}")
        sys.exit(1)

    # Check if patch was already applied
    check_result = subprocess.run(
        ["git", "apply", "--check", PATCH_FILE],
        cwd=LLAMA_CPP_DIR,
        capture_output=True,
    )
    if check_result.returncode == 0:
        run(
            ["git", "apply", PATCH_FILE],
            cwd=LLAMA_CPP_DIR,
            description="Applying TQ3_0/TQ2_0 patch...",
        )
    else:
        print("    Patch appears already applied or conflicting — skipping.")

    # --- CMake configure ---
    step("Configure with CMake (Vulkan)")
    build_dir = os.path.join(LLAMA_CPP_DIR, "build")
    run(
        ["cmake", "-B", "build", "-DGGML_VULKAN=ON", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=LLAMA_CPP_DIR,
        description="Configuring...",
    )

    # --- CMake build ---
    step("Build llama.cpp")
    if IS_WINDOWS:
        run(
            ["cmake", "--build", "build", "--config", "Release", "--parallel"],
            cwd=LLAMA_CPP_DIR,
            description="Building (Release)...",
        )
    else:
        cpu_count = os.cpu_count() or 4
        run(
            ["cmake", "--build", "build", "--", f"-j{cpu_count}"],
            cwd=LLAMA_CPP_DIR,
            description=f"Building with {cpu_count} threads...",
        )

    # --- Find llama-server ---
    step("Locate llama-server binary")
    candidates = [
        os.path.join(build_dir, "bin", "Release", "llama-server.exe"),
        os.path.join(build_dir, "bin", "llama-server.exe"),
        os.path.join(build_dir, "bin", "Release", "llama-server"),
        os.path.join(build_dir, "bin", "llama-server"),
    ]
    server_path = None
    for path in candidates:
        if os.path.isfile(path):
            server_path = path
            break

    if server_path:
        print(f"    Found: {server_path}")
    else:
        print("    WARNING: llama-server not found in expected locations.")
        print("    Build may have succeeded — check the build/ directory manually.")

    # --- Create models/ ---
    step("Create models/ directory")
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"    Ready: {MODELS_DIR}")

    # --- Done ---
    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Download a GGUF model and place it in:")
    print(f"       {MODELS_DIR}")
    print()
    print("  2. Launch the interactive server:")
    print("       python scripts/launcher.py")
    print()
    if server_path:
        print("  Or run manually:")
        print(f"    {server_path} \\")
        print("      -m models/YOUR_MODEL.gguf -ngl 30 -ctk tq3_0 -ctv tq3_0")
    print()


if __name__ == "__main__":
    main()
