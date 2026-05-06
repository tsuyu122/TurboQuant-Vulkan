"""TurboQuant v3.3 — real architectural cold/hot KV split.

Phase 1 (this package): Python reference implementation and empirical study
of the cold V-cache format + fallback gate. All numeric constants used in
the C/Vulkan port come from ``tq3_3_study.py`` measurements, not hand-tuned
guesses.

Phase 2 (llama.cpp KV cache): hot/cold region split in
``llama-kv-cache-unified`` with per-cell demotion + fallback markers.

Phase 3 (Vulkan FA kernel): dual-region shader reading cold bulk in the new
18 B / 64-element format, hot tail in existing TQ2_0/TQ3_2 format.

See ``ROADMAP.md`` for exact file-level entry points.
"""
