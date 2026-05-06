# Contributing to TurboQuant Vulkan

Thanks for your interest in contributing.

This is a hobby project — I built it to learn and experiment with KV cache quantization on Vulkan. I don't plan to maintain it actively or provide regular updates, but contributions are welcome.

## Before You Start

- This project is a **patch for llama.cpp** (commit `073bb2c`, tag `b8762`). Make sure your changes apply cleanly against that base.
- The implementation is **pure C + GLSL** — no Python/CUDA dependencies in the core code.
- The benchmark scripts are in Python and use the llama.cpp server API.

## Pull Requests

- Keep each PR focused on one logical change
- Explain what changed and why
- Test with the Vulkan backend before submitting
- If you touch shader code, verify with at least one quantized inference run

## Reporting Bugs

Open an issue on GitHub with:

- Reproduction steps
- Expected result
- Actual result
- GPU model and driver version
- Logs or screenshots when useful

## Code Style

- C code: match the existing llama.cpp/ggml style
- GLSL shaders: match the existing Vulkan shader conventions in llama.cpp
- Keep changes minimal and focused

## License

By contributing, you agree that your work is licensed under the [GNU General Public License v3.0](LICENSE).
