"""Generate an LLM-judge-friendly markdown report from runner output."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent


JUDGE_HEADER = """\
# TurboQuant KV-Cache Benchmark — Judge Workbook

You are a reviewer grading the outputs of several KV-cache quantization
configurations of the same underlying model. All configurations answered the
same 100 prompts; only the KV-cache precision differs.

**Task.** For each prompt, read every configuration's response and score it
on a 1–10 scale using the rubric below. Then write a one-sentence justification.

**Rubric (apply per response):**
- **10** — fully correct, well organised, no hallucinations.
- **8–9** — correct with minor style or completeness issues.
- **6–7** — mostly correct but with notable gaps or mild errors.
- **4–5** — partially correct; significant errors or omissions.
- **2–3** — mostly wrong / incoherent.
- **1** — empty, off-topic, or degenerate (repetition / gibberish).

Special flags:
- `OOM_SKIPPED` means the server could not host the prompt's context tier
  — do not penalise, simply note it.
- `ERROR: ...` means the HTTP call failed — score 1.

Prompts are grouped by **category** and then by **context tier** (token
budget). Within each cell, all configurations' answers are shown side-by-side
so you can compare them directly.

---
"""


def _fmt_meta(r: dict[str, Any]) -> str:
    if r.get("oom_skipped"):
        return f"*(OOM_SKIPPED — {r.get('error','')})*"
    if r.get("error"):
        return f"*(ERROR: {r['error']})*"
    return (f"*({r['tokens_out']} tok · {r['tokens_per_s']:.2f} t/s · "
            f"{r['elapsed_s']:.1f} s)*")


def _fmt_response(res: dict) -> str:
    parts: list[str] = []
    reasoning = (res.get("reasoning") or "").rstrip()
    answer    = (res.get("response") or "").rstrip()
    if reasoning:
        parts.append("<details><summary>reasoning</summary>\n\n```\n" + reasoning + "\n```\n\n</details>")
    if answer:
        parts.append("**Answer:**\n\n```\n" + answer + "\n```")
    elif not reasoning:
        return "_(empty)_"
    elif not answer:
        parts.append("**Answer:** _(empty — finish_reason=" + (res.get("finish_reason") or "?") + ")_")
    return "\n\n".join(parts)


def build_report(raw: dict[str, Any]) -> str:
    runs = raw["runs"]
    labels = [r["label"] for r in runs]

    # Index responses by (config_label, prompt_id).
    by_config: dict[str, dict[str, dict]] = {
        r["label"]: {res["id"]: res for res in r["results"]} for r in runs
    }

    # Gather unique prompts in order (using first run).
    seen_ids: list[str] = []
    prompt_by_id: dict[str, dict] = {}
    for r in runs:
        for res in r["results"]:
            if res["id"] not in prompt_by_id:
                prompt_by_id[res["id"]] = res
                seen_ids.append(res["id"])

    # Group by (category, ctx_tier).
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for pid in seen_ids:
        p = prompt_by_id[pid]
        grouped[(p["category"], p["ctx_tier"])].append(pid)

    lines: list[str] = [JUDGE_HEADER]

    # Summary table.
    lines.append("## Run metadata\n")
    lines.append(f"- Model: `{raw['model']}`")
    lines.append(f"- ngl:   {raw['ngl']}")
    lines.append(f"- Total prompts: {raw['prompts']}")
    lines.append("")
    lines.append("| Config | ctk | ctv | Loaded ctx | Failed launch |")
    lines.append("|--------|-----|-----|------------|---------------|")
    for r in runs:
        lines.append(f"| `{r['label']}` | `{r['ctk']}` | `{r['ctv']}` | "
                     f"{r['loaded_ctx']} | {'yes' if r['failed_launch'] else 'no'} |")
    lines.append("")

    # Per-category, per-tier sections.
    cats_order  = ["math", "logic", "reasoning", "coding", "knowledge"]
    tiers_order = [128, 256, 512, 1_000_000, 2_000_000]

    for cat in cats_order:
        lines.append(f"## Category: {cat}\n")
        for tier in tiers_order:
            ids = grouped.get((cat, tier), [])
            if not ids:
                continue
            lines.append(f"### ctx tier = {tier}\n")
            for pid in ids:
                p = prompt_by_id[pid]
                lines.append(f"#### `{pid}`")
                lines.append("")
                lines.append("**Prompt:**")
                lines.append("")
                lines.append("```")
                lines.append(p["prompt"])
                lines.append("```")
                lines.append("")
                for lab in labels:
                    res = by_config[lab].get(pid)
                    if res is None:
                        lines.append(f"**`{lab}`** — _(no data)_")
                        lines.append("")
                        continue
                    lines.append(f"**`{lab}`** {_fmt_meta(res)}")
                    lines.append("")
                    lines.append(_fmt_response(res))
                    lines.append("")
                    lines.append(f"_Score ({lab}): ___ / 10 — Notes: _")
                    lines.append("")
                lines.append("---")
                lines.append("")

    # Aggregate performance table.
    lines.append("## Aggregate performance\n")
    lines.append("| Config | Avg t/s | Median t/s | Avg latency (s) | Completed | Errors | OOM-skipped |")
    lines.append("|--------|---------|------------|-----------------|-----------|--------|-------------|")
    for r in runs:
        tps = [x["tokens_per_s"] for x in r["results"] if not x["oom_skipped"] and not x["error"]]
        lat = [x["elapsed_s"]     for x in r["results"] if not x["oom_skipped"] and not x["error"]]
        err = sum(1 for x in r["results"] if x["error"] and not x["oom_skipped"])
        oom = sum(1 for x in r["results"] if x["oom_skipped"])
        done = len(tps)
        if tps:
            tps_s = sorted(tps)
            avg_tps = sum(tps)/len(tps)
            med_tps = tps_s[len(tps_s)//2]
            avg_lat = sum(lat)/len(lat)
            lines.append(f"| `{r['label']}` | {avg_tps:.2f} | {med_tps:.2f} | {avg_lat:.2f} | "
                         f"{done} | {err} | {oom} |")
        else:
            lines.append(f"| `{r['label']}` | — | — | — | 0 | {err} | {oom} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="bench_suite/results/raw.json")
    ap.add_argument("--out", default="bench_suite/results/RESULTS.md")
    args = ap.parse_args()

    raw_path = (ROOT / args.raw).resolve()
    out_path = (ROOT / args.out).resolve()
    if not raw_path.exists():
        print(f"raw results not found: {raw_path}")
        return 1
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(raw), encoding="utf-8")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
