# bench_suite — TurboQuant KV-cache benchmark

Sequential head-to-head evaluation of five KV-cache precisions on the same
underlying model, using a fixed 100-prompt corpus and a separate LLM as
judge.

## What it measures

For each KV-cache configuration the suite:

1. Launches `llama-server` **once** (no reload between prompts).
2. Runs all 100 prompts sequentially and collects the response text,
   token count, and throughput.
3. Shuts the server down cleanly and moves to the next configuration.

KV configurations (`-ctk` / `-ctv`):

| Label    | ctk     | ctv     | Notes                              |
|----------|---------|---------|------------------------------------|
| `f16`    | f16     | f16     | baseline (no KV quantisation)      |
| `tq2_0`  | tq2_0   | tq2_0   | 2-bit both                          |
| `tq3_0`  | tq3_0   | tq3_0   | TurboQuant v1                       |
| `tq3_1`  | tq3_0   | tq2_0   | published as TQ3_1 (mixed KV)       |
| `tq3_2`  | tq3_0   | tq3_2   | TurboQuant v2 (64-block V)          |

## Prompt design

- **5 categories** — math, logic, reasoning, coding, knowledge
- **5 context tiers** — 128 / 256 / 512 / 1 000 000 / 2 000 000 tokens
- **4 prompts per cell** → 100 total

Each prompt carries an `n_predict` budget matched to its tier. Prompts in
a tier larger than the server's loaded `-c` are auto-skipped and tagged
`OOM_SKIPPED` so the judge can compare apples-to-apples.

## OOM / VRAM fallback

Each configuration is attempted at `c = 2 000 000` first and, on launch
failure, falls back through `1 000 000 → 524 288 → 262 144 → 131 072 → 65 536 →
32 768 → 16 384 → 8 192 → 4 096 → 2 048 → 1 024 → 512 → 256 → 128`. The run for
that configuration proceeds at the largest context the GPU would accept.
Prompts whose tier exceeds the loaded context are auto-skipped
(`OOM_SKIPPED`) so every row in the report is still present.

## How to run

### Full run (~ hours, 5 configs × 100 prompts)

```
py -m bench_suite.run_all
```

### Smoke test (1 config, 2 prompts)

```
py -m bench_suite.runner --configs tq3_0 --limit 2
py -m bench_suite.report
```

### Useful options

```
--configs f16 tq3_0 tq3_2      # subset of KV configs
--ngl 24                       # GPU layer offload
--model PATH                   # alternate GGUF
--port 8899                    # server port
--limit N                      # run first N prompts only
--timeout 600                  # per-prompt HTTP timeout
```

## Output

- `bench_suite/results/raw.json`   — per-prompt results for every config (re-written after each model so partial runs are preserved).
- `bench_suite/results/RESULTS.md` — human/LLM-judge workbook with prompt + every config's response side-by-side and empty score slots.
- `bench_suite/results/server_logs/` — one stderr log per server launch attempt (useful when OOM fallback triggers).

## Judge workflow

1. Run the bench.
2. Open `RESULTS.md` in another LLM (or paste section-by-section).
3. The judge fills in `Score (<config>): __ / 10` for each response using
   the rubric included at the top of the file.
4. Aggregate the scores per category/tier to compare configurations.

## Files

| File                | Role                                                     |
|---------------------|----------------------------------------------------------|
| `prompts.py`        | 100-prompt dataset                                        |
| `runner.py`         | launches server, runs suite, writes `raw.json`            |
| `report.py`         | renders `raw.json` into `RESULTS.md`                      |
| `run_all.py`        | one-shot end-to-end entry point                           |
