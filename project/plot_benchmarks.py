#!/usr/bin/env python3
"""Generate lightweight benchmark summaries and SVG plots.

This script intentionally uses only the Python standard library so it can run on
Clariden login nodes without extra plotting dependencies.
"""

import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LOG = ROOT / "project" / "benchmark_log.csv"
OUT_DIR = ROOT / "project" / "plots"
SUMMARY_CSV = OUT_DIR / "throughput_summary.csv"
NORMAL_SVG = OUT_DIR / "760m_normal_throughput.svg"


def get_partition(run):
    text = "{} {}".format(run["feature"], run["notes"]).lower()
    if "debug" in text:
        return "debug"
    if "normal" in text:
        return "normal"
    return "unknown"


def load_runs():
    runs = []
    with BENCHMARK_LOG.open(newline="") as f:
        for row in csv.DictReader(f):
            token_text = row.get("tokens_per_sec_per_gpu", "").strip()
            if not token_text or row.get("status") != "completed":
                continue
            try:
                run = {
                    "date": row["date"],
                    "feature": row["feature"],
                    "model_size": row["model_size"],
                    "mode": row["mode"],
                    "nodes": int(row["nodes"]),
                    "gpus": int(row["gpus"]),
                    "mbs": int(row["micro_batch_size"]),
                    "gbs": int(row["global_batch_size"]),
                    "steps": int(row["steps"]),
                    "tokens": float(token_text),
                    "status": row["status"],
                    "log_path": row["log_path"],
                    "notes": row["notes"],
                }
                run["partition"] = get_partition(run)
                runs.append(run)
            except (KeyError, ValueError):
                continue
    return runs


def write_summary(runs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date",
            "model_size",
            "partition",
            "feature",
            "nodes",
            "gpus",
            "micro_batch_size",
            "global_batch_size",
            "steps",
            "tokens_per_sec_per_gpu",
            "log_path",
        ])
        for run in runs:
            tokens = run["tokens"]
            writer.writerow([
                run["date"],
                run["model_size"],
                run["partition"],
                run["feature"],
                run["nodes"],
                run["gpus"],
                run["mbs"],
                run["gbs"],
                run["steps"],
                int(tokens) if tokens.is_integer() else round(tokens, 1),
                run["log_path"],
            ])


def svg_escape(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_bar_svg(runs):
    wanted = set([
        "distributed-optimizer-normal-20s-baseline",
        "no-distributed-optimizer-normal-20s",
        "mbs2-normal-20s",
        "mbs8-normal-20s",
        "normal-baseline-50s",
    ])
    selected = [
        r for r in runs
        if r["model_size"] == "760m"
        and r["nodes"] == 1
        and r["partition"] == "normal"
        and r["feature"] in wanted
    ]
    order = {
        "mbs2-normal-20s": 0,
        "distributed-optimizer-normal-20s-baseline": 1,
        "no-distributed-optimizer-normal-20s": 2,
        "mbs8-normal-20s": 3,
        "normal-baseline-50s": 4,
    }
    selected.sort(key=lambda r: order.get(r["feature"], 99))
    if not selected:
        NORMAL_SVG.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n")
        return

    labels = {
        "mbs2-normal-20s": "MBS 2",
        "distributed-optimizer-normal-20s-baseline": "MBS 4 baseline",
        "no-distributed-optimizer-normal-20s": "No dist opt",
        "mbs8-normal-20s": "MBS 8",
        "normal-baseline-50s": "MBS 4, 50 steps",
    }
    max_value = max(r["tokens"] for r in selected) * 1.15
    width = 980
    height = 420
    left = 190
    right = 40
    top = 68
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_gap = 16
    bar_h = (plot_h - bar_gap * (len(selected) - 1)) / len(selected)

    palette = ["#2563eb", "#059669", "#64748b", "#d97706", "#7c3aed"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">760m normal throughput comparison</text>',
        '<text x="40" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Final-iteration tokens/sec/GPU from completed runs; use notes for stable-window averages.</text>',
    ]

    tick = 0
    while tick <= int(max_value) + 1:
        x = left + tick / max_value * plot_w
        lines.append('<line x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}" stroke="#e5e7eb" stroke-width="1"/>'.format(x, top, x, height-bottom))
        lines.append('<text x="{:.1f}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{}k</text>'.format(x, height-bottom+24, tick//1000))
        tick += 10000

    for idx, run in enumerate(selected):
        y = top + idx * (bar_h + bar_gap)
        bar_w = run["tokens"] / max_value * plot_w
        color = palette[idx % len(palette)]
        label = labels.get(run["feature"], run["feature"])
        lines.append('<text x="{}" y="{:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#111827">{}</text>'.format(left-14, y + bar_h/2 + 4, svg_escape(label)))
        lines.append('<rect x="{}" y="{:.1f}" width="{:.1f}" height="{:.1f}" rx="4" fill="{}"/>'.format(left, y, bar_w, bar_h, color))
        lines.append('<text x="{:.1f}" y="{:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{:,}</text>'.format(left + bar_w + 8, y + bar_h/2 + 4, int(run["tokens"])))

    normal_760 = [r["tokens"] for r in runs if r["model_size"] == "760m" and r["partition"] == "normal"]
    if normal_760:
        lines.append('<text x="40" y="{}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">Normal 760m completed-run mean final throughput: {:.0f} tokens/sec/GPU</text>'.format(height-24, mean(normal_760)))
    lines.append("</svg>")
    NORMAL_SVG.write_text("\n".join(lines) + "\n")


def main():
    runs = load_runs()
    write_summary(runs)
    write_bar_svg(runs)
    print("Wrote {}".format(SUMMARY_CSV))
    print("Wrote {}".format(NORMAL_SVG))


if __name__ == "__main__":
    main()
