#!/usr/bin/env python3
"""Generate benchmark summaries and simple SVG plots from raw logs.

The benchmark log records the run inventory. This script makes the comparison
more defensible by parsing raw iteration throughput from each available log and
reporting stable-window metrics separately from final-iteration values.
It intentionally uses only the Python standard library for Clariden login nodes.
"""

import csv
import re
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LOG = ROOT / "project" / "benchmark_log.csv"
OUT_DIR = ROOT / "project" / "plots"
SUMMARY_CSV = OUT_DIR / "throughput_summary.csv"
DEFENSIBLE_CSV = OUT_DIR / "defensible_throughput_summary.csv"
NORMAL_SVG = OUT_DIR / "760m_normal_stable_throughput.svg"
DEBUG_SVG = OUT_DIR / "760m_debug_stable_throughput.svg"
LEGACY_NORMAL_SVG = OUT_DIR / "760m_normal_throughput.svg"

CLAIM_FEATURES = set([
    "distributed-optimizer-normal-20s-baseline",
    "no-distributed-optimizer-normal-20s",
    "mbs2-normal-20s",
    "mbs8-normal-20s",
    "normal-baseline-50s",
])

DEBUG_FEATURES = set([
    "debug-setup-baseline",
    "post-merge-debug-sanity",
    "no-distributed-optimizer-debug",
    "nsys-first-profile",
])

LABELS = {
    "debug-setup-baseline": "Old debug setup",
    "post-merge-debug-sanity": "Debug baseline",
    "no-distributed-optimizer-debug": "No dist opt debug",
    "nsys-first-profile": "Debug NSYS",
    "mbs2-normal-20s": "MBS 2",
    "distributed-optimizer-normal-20s-baseline": "MBS 4 baseline",
    "no-distributed-optimizer-normal-20s": "No dist opt",
    "mbs8-normal-20s": "MBS 8",
    "normal-baseline-50s": "MBS 4, 50 steps",
    "nsys-normal-profile-timeout": "NSYS timeout",
}

NORMAL_ORDER = {
    "mbs2-normal-20s": 0,
    "distributed-optimizer-normal-20s-baseline": 1,
    "no-distributed-optimizer-normal-20s": 2,
    "mbs8-normal-20s": 3,
    "normal-baseline-50s": 4,
}

DEBUG_ORDER = {
    "debug-setup-baseline": 0,
    "post-merge-debug-sanity": 1,
    "no-distributed-optimizer-debug": 2,
    "nsys-first-profile": 3,
}


def get_partition(row):
    text = "{} {}".format(row.get("feature", ""), row.get("notes", "")).lower()
    if "debug" in text:
        return "debug"
    if "normal" in text:
        return "normal"
    if row.get("feature") == "nsys-first-profile":
        return "debug"
    return "unknown"


def parse_int(text, default=None):
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def parse_float(text, default=None):
    try:
        if text is None or text == "":
            return default
        return float(text)
    except ValueError:
        return default


def parse_iterations(log_path):
    if not log_path:
        return []
    path = ROOT / log_path
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    # srun can wrap long lines as "... per GP\n0: U ...". Search over the
    # whole log so wrapped throughput lines still parse.
    pattern = re.compile(
        r"iteration\s+(\d+)\s*/\s*(\d+).*?tokens/sec/GPU:\s*(\d+)",
        re.DOTALL,
    )
    iterations = []
    seen = set()
    for match in pattern.finditer(text):
        iteration = int(match.group(1))
        total = int(match.group(2))
        tokens = int(match.group(3))
        key = (iteration, total)
        if key in seen:
            continue
        seen.add(key)
        iterations.append({"iteration": iteration, "total": total, "tokens": tokens})
    iterations.sort(key=lambda item: item["iteration"])
    return iterations


def summarize_iterations(iterations):
    if not iterations:
        return {
            "parsed_iterations": 0,
            "parsed_final": "",
            "avg_excluding_first2": "",
            "median_excluding_first2": "",
            "min_excluding_first2": "",
            "max_excluding_first2": "",
            "avg_last_half": "",
            "median_last_half": "",
            "defensible_metric": "",
            "defensible_metric_name": "",
            "iteration_values": "",
        }
    values = [item["tokens"] for item in iterations]
    total = iterations[-1]["total"]
    warm_values = [item["tokens"] for item in iterations if item["iteration"] >= 3]
    if not warm_values:
        warm_values = values[:]
    half_start = max(1, total // 2 + 1)
    last_half_values = [item["tokens"] for item in iterations if item["iteration"] >= half_start]
    if not last_half_values:
        last_half_values = warm_values[:]
    if len(iterations) >= 20:
        metric_values = last_half_values
        metric_name = "avg_last_half"
    else:
        metric_values = warm_values
        metric_name = "avg_excluding_first2"
    return {
        "parsed_iterations": len(iterations),
        "parsed_final": values[-1],
        "avg_excluding_first2": round(mean(warm_values)),
        "median_excluding_first2": round(median(warm_values)),
        "min_excluding_first2": min(warm_values),
        "max_excluding_first2": max(warm_values),
        "avg_last_half": round(mean(last_half_values)),
        "median_last_half": round(median(last_half_values)),
        "defensible_metric": round(mean(metric_values)),
        "defensible_metric_name": metric_name,
        "iteration_values": ";".join(["{}:{}".format(item["iteration"], item["tokens"]) for item in iterations]),
    }


def load_runs():
    runs = []
    with BENCHMARK_LOG.open(newline="") as f:
        for row in csv.DictReader(f):
            run = dict(row)
            run["partition"] = get_partition(run)
            run["nodes_int"] = parse_int(run.get("nodes"), 0)
            run["gpus_int"] = parse_int(run.get("gpus"), 0)
            run["mbs_int"] = parse_int(run.get("micro_batch_size"), 0)
            run["gbs_int"] = parse_int(run.get("global_batch_size"), 0)
            run["steps_int"] = parse_int(run.get("steps"), 0)
            run["logged_tokens"] = parse_float(run.get("tokens_per_sec_per_gpu"), None)
            iterations = parse_iterations(run.get("log_path", ""))
            run.update(summarize_iterations(iterations))
            if run["defensible_metric"] == "" and run["logged_tokens"] is not None:
                run["defensible_metric"] = round(run["logged_tokens"])
                run["defensible_metric_name"] = "logged_tokens_per_sec_per_gpu"
            run["usable_for_claims"] = "yes" if run.get("status") == "completed" and run["partition"] == "normal" and run["parsed_iterations"] else "no"
            if run.get("status") in ("failed", "timeout", "cancelled", "submitted"):
                run["usable_for_claims"] = "no"
            runs.append(run)
    return runs


def display_number(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return int(value) if value.is_integer() else round(value, 1)
    return value


def write_summaries(runs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basic_fields = [
        "date", "model_size", "partition", "feature", "status", "nodes", "gpus",
        "micro_batch_size", "global_batch_size", "steps", "tokens_per_sec_per_gpu",
        "log_path",
    ]
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(basic_fields)
        for run in runs:
            if not run.get("tokens_per_sec_per_gpu"):
                continue
            writer.writerow([run.get(field, "") for field in basic_fields])

    detailed_fields = [
        "date",
        "model_size",
        "partition",
        "feature",
        "status",
        "usable_for_claims",
        "nodes",
        "gpus",
        "micro_batch_size",
        "global_batch_size",
        "steps",
        "logged_tokens_per_sec_per_gpu",
        "parsed_iterations",
        "parsed_final",
        "avg_excluding_first2",
        "median_excluding_first2",
        "min_excluding_first2",
        "max_excluding_first2",
        "avg_last_half",
        "median_last_half",
        "defensible_metric_name",
        "defensible_metric",
        "log_path",
        "iteration_values",
    ]
    with DEFENSIBLE_CSV.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(detailed_fields)
        for run in runs:
            writer.writerow([
                run.get("date", ""),
                run.get("model_size", ""),
                run.get("partition", ""),
                run.get("feature", ""),
                run.get("status", ""),
                run.get("usable_for_claims", ""),
                run.get("nodes", ""),
                run.get("gpus", ""),
                run.get("micro_batch_size", ""),
                run.get("global_batch_size", ""),
                run.get("steps", ""),
                display_number(run.get("logged_tokens")),
                run.get("parsed_iterations", ""),
                display_number(run.get("parsed_final")),
                display_number(run.get("avg_excluding_first2")),
                display_number(run.get("median_excluding_first2")),
                display_number(run.get("min_excluding_first2")),
                display_number(run.get("max_excluding_first2")),
                display_number(run.get("avg_last_half")),
                display_number(run.get("median_last_half")),
                run.get("defensible_metric_name", ""),
                display_number(run.get("defensible_metric")),
                run.get("log_path", ""),
                run.get("iteration_values", ""),
            ])


def svg_escape(text):
    return (
        str(text).replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_bar_svg(path, title, subtitle, runs, order, footer):
    selected = [run for run in runs if run.get("defensible_metric") not in ("", None)]
    selected.sort(key=lambda run: order.get(run.get("feature"), 99))
    if not selected:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n")
        return
    max_value = max(float(run["defensible_metric"]) for run in selected) * 1.18
    width = 1080
    height = max(420, 112 + len(selected) * 58)
    left = 230
    right = 110
    top = 76
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_gap = 16
    bar_h = (plot_h - bar_gap * (len(selected) - 1)) / len(selected)
    palette = ["#2563eb", "#059669", "#64748b", "#d97706", "#7c3aed", "#0891b2"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{}</text>'.format(svg_escape(title)),
        '<text x="40" y="56" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{}</text>'.format(svg_escape(subtitle)),
    ]
    tick = 0
    tick_step = 10000
    while tick <= int(max_value) + tick_step:
        x = left + tick / max_value * plot_w
        if x <= left + plot_w:
            lines.append('<line x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}" stroke="#e5e7eb" stroke-width="1"/>'.format(x, top, x, height-bottom))
            lines.append('<text x="{:.1f}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#6b7280">{}k</text>'.format(x, height-bottom+24, tick//1000))
        tick += tick_step
    for idx, run in enumerate(selected):
        y = top + idx * (bar_h + bar_gap)
        value = float(run["defensible_metric"])
        bar_w = value / max_value * plot_w
        color = palette[idx % len(palette)]
        label = LABELS.get(run.get("feature"), run.get("feature"))
        detail = "{}; {} iters".format(run.get("defensible_metric_name", "metric"), run.get("parsed_iterations", "?"))
        lines.append('<text x="{}" y="{:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#111827">{}</text>'.format(left-14, y + bar_h/2 - 3, svg_escape(label)))
        lines.append('<text x="{}" y="{:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#6b7280">{}</text>'.format(left-14, y + bar_h/2 + 12, svg_escape(detail)))
        lines.append('<rect x="{}" y="{:.1f}" width="{:.1f}" height="{:.1f}" rx="4" fill="{}"/>'.format(left, y, bar_w, bar_h, color))
        lines.append('<text x="{:.1f}" y="{:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{:,}</text>'.format(left + bar_w + 8, y + bar_h/2 + 4, int(round(value))))
    lines.append('<text x="40" y="{}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{}</text>'.format(height-24, svg_escape(footer)))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_plots(runs):
    normal = [
        run for run in runs
        if run.get("feature") in CLAIM_FEATURES
        and run.get("model_size") == "760m"
        and run.get("partition") == "normal"
        and run.get("status") == "completed"
        and run.get("parsed_iterations")
    ]
    debug = [
        run for run in runs
        if run.get("feature") in DEBUG_FEATURES
        and run.get("model_size") == "760m"
        and run.get("partition") == "debug"
        and run.get("status") == "completed"
        and run.get("parsed_iterations")
    ]
    write_bar_svg(
        NORMAL_SVG,
        "760m normal throughput, stable-window metrics",
        "Bars use log-parsed warmup-excluded averages, not single final iterations.",
        normal,
        NORMAL_ORDER,
        "Use this chart for normal-partition claims; timeout/failed/debug runs are excluded.",
    )
    # Keep the previous filename working for slide references.
    LEGACY_NORMAL_SVG.write_text(NORMAL_SVG.read_text())
    write_bar_svg(
        DEBUG_SVG,
        "760m debug throughput, directional only",
        "Debug partition runs are separated because they are not comparable to normal partition runs.",
        debug,
        DEBUG_ORDER,
        "Use debug data only for smoke tests and directional hypotheses, not final throughput claims.",
    )


def main():
    runs = load_runs()
    write_summaries(runs)
    write_plots(runs)
    print("Wrote {}".format(SUMMARY_CSV))
    print("Wrote {}".format(DEFENSIBLE_CSV))
    print("Wrote {}".format(NORMAL_SVG))
    print("Wrote {}".format(DEBUG_SVG))


if __name__ == "__main__":
    main()
