from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Sequence
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch


REPORT_TITLE = "Blase observability Report"


SHORT_WORKER_LABELS = {
    "sheet_detection": "Sheet det.",
    "recognition": "Recognition",
    "grading": "Grading",
    "written": "Written",
    "detection_worker": "Detection",
    "complete_worker": "Complete",
    "written_worker": "Written",
    "Grading_worker": "Grading",
    "question_generation": "Q-Gen",
}

WORKER_STYLE: dict[str, tuple[str, str, str]] = {
    "sheet_detection": ("Sheet Detection", "#79C0FF", "#1F3A5F"),
    "recognition": ("Recognition", "#58A6FF", "#264A73"),
    "grading": ("Grading", "#F0883E", "#6B3F1E"),
    "written": ("Written", "#BC8CFF", "#5A3F8A"),
    "written_extraction": ("Text Extraction", "#79C0FF", "#1F3A5F"),
    "written_grading": ("Written Grading", "#BC8CFF", "#5A3F8A"),
    "choices_grading": ("Choices Grading", "#F0883E", "#6B3F1E"),
    "question_gen": ("Question Generation", "#3FB950", "#1A3D24"),
    "detection_worker": ("Sheet Detection", "#79C0FF", "#1F3A5F"),
    "complete_worker": ("Complete Worker", "#58A6FF", "#264A73"),
    "written_worker": ("Written Worker", "#BC8CFF", "#5A3F8A"),
    "Grading_worker": ("Grading Worker", "#F0883E", "#6B3F1E"),
    "question_generation": ("Question Generation", "#3FB950", "#1A3D24"),
}

THEME: dict[str, str] = {
    
"page_bg": "#0D1117", "card_bg": "#161B22", "card_border": "#30363D", "chart_bg": "#0D1117", "text": "#E6EDF3", "text_muted": "#8B949E", "grid": "#21262D", "tick": "#6E7681", "cm_correct": "#238636", "cm_error": "#DA3633", "cm_mid": "#21262D", "pie_ok": "#3FB950", "pie_fail": "#F85149", "pie_warn": "#D29922", "token_secondary": "#79C0FF", "token_in": "#58A6FF", "api_line": "#F0883E", "perf_track": "#21262D", "cm_tp": "#238636", "cm_tn": "#1F6FEB", "cm_fp": "#DA3633", "cm_fn": "#DB6D28",
}


def worker_style(key: str) -> tuple[str, str, str]:
    return WORKER_STYLE.get(key, (key, "#8B949E", "#21262D"))


def _to_binary(value: float) -> int:
    return 1 if float(value) >= 0.5 else 0


def match_accuracy(pairs: Sequence[dict], status: str) -> tuple[int, int, float]:
    """
    Compute (correct, total, accuracy_pct) for a list of
    {"predicted": ..., "ground_truth": ...} dicts.

    status="MCQ"      — compare single-char letters (strip whitespace)
    status="complete" — compare each character of the string
    status="written"  — same as complete (per-character on stripped strings)
    """
    if not pairs:
        return 0, 0, 0.0

    if status in ("complete", "written"):
        from itertools import zip_longest
        correct = 0
        total = 0
        for p in pairs:
            gt = "".join(str(p["ground_truth"]).split())
            pred = "".join(str(p["predicted"]).split())
            for g, pr in zip_longest(gt, pred, fillvalue=None):
                total += 1
                if g == pr:
                    correct += 1
        return correct, total, 100.0 * correct / total if total else 0.0
    # MCQ / default — exact match per entry
    correct = sum(
        1 for p in pairs
        if str(p["ground_truth"]).strip() == str(p["predicted"]).strip()
    )
    return correct, len(pairs), correct / len(pairs) * 100


def binary_confusion(ground_truth: Sequence[float], predicted: Sequence[float]) -> dict:
    tp = tn = fp = fn = 0
    for gt, pr in zip(ground_truth, predicted):
        g, p = _to_binary(gt), _to_binary(pr)
        if   g == 1 and p == 1: tp += 1
        elif g == 0 and p == 0: tn += 1
        elif g == 0 and p == 1: fp += 1
        else:                   fn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100 if total else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total, "accuracy": accuracy}


def confusion_matrix(conf: dict) -> np.ndarray:
    return np.array([[conf["tp"], conf["fn"]], [conf["fp"], conf["tn"]]], dtype=int)


def grade_stats(comp: dict) -> dict:
    gt = np.array(comp["ground_truth"], dtype=float)
    pr = np.array(comp["predicted"],    dtype=float)
    if gt.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "r": 0.0}
    mae  = float(np.mean(np.abs(gt - pr)))
    rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))
    if gt.size >= 2 and gt.std() > 0 and pr.std() > 0:
        r = float(np.corrcoef(gt, pr)[0, 1])
    else:
        r = 0.0
    return {"mae": mae, "rmse": rmse, "r": r}


def within_tolerance(comp: dict, tol: float) -> float:
    gt = np.array(comp["ground_truth"], dtype=float)
    pr = np.array(comp["predicted"],    dtype=float)
    if gt.size == 0:
        return 0.0
    return float(np.mean(np.abs(gt - pr) <= tol) * 100)


def usage_totals(usage: dict) -> dict:
    units = usage.get("units", [])
    tokens_input    = sum(int(u.get("tokens_input",  0)) for u in units)
    tokens_output   = sum(int(u.get("tokens_output", 0)) for u in units)
    api_calls_total = sum(int(u.get("api_calls",     0)) for u in units)
    return {
        "tokens_input":    tokens_input,
        "tokens_output":   tokens_output,
        "tokens_total":    tokens_input + tokens_output,
        "api_calls_total": api_calls_total,
    }


def normalize_usage(raw: dict) -> dict:
    if "units" in raw:
        return raw
    out = dict(raw)
    if "unit" in out:
        out["units"] = [out.pop("unit")]
    else:
        out["units"] = []
    return out


def build_question_gen_usage(question_generation: list[dict]) -> dict:
    all_units: list[dict] = []
    unit_label = "chunk"
    for rec in question_generation:
        metrics    = rec.get("metrics", {})
        unit_label = metrics.get("unit_label", unit_label)
        for u in metrics.get("units", []):
            all_units.append(u)
    return {"unit_label": unit_label, "units": all_units}


def load_report_data(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    required_top = {
        "detection_worker", "complete_worker", "written_worker",
        "Grading_worker", "question_generation",
        "recognition", "grading",
    }
    if missing := required_top - set(data.keys()):
        raise ValueError(f"Report JSON missing top-level keys: {sorted(missing)}")
    return data

def build_worker_metrics(name, records):
    if not records:
        return {
            "worker": name,
            "throughput_peak": 0,
            "throughput_mean": 0,
            "throughput_avg": 0,
            "duration_peak": 0,
            "duration_mean": 0,
            "duration_avg": 0,
            "memory_peak_mb": 0,
            "memory_mean_mb": 0,
            "memory_avg_mb": 0,
        }

    durations = [r["duration_ms"] for r in records if r.get("duration_ms", 0) > 0]
    throughputs = [(1000.0 * 60) / d for d in durations] if durations else [0]
    memories = [r.get("ram_mb", 0) for r in records]
    durations_s = [d / 1000 for d in durations]

    return {
        "worker": name,

        "throughput_peak": max(throughputs),
        "throughput_mean": float(np.mean(throughputs)),
        "throughput_avg": float(np.median(throughputs)),

        "duration_peak": max(durations_s),
        "duration_mean": float(np.mean(durations_s)),
        "duration_avg": float(np.median(durations_s)),

        "memory_peak_mb": max(memories),
        "memory_mean_mb": float(np.mean(memories)),
        "memory_avg_mb": float(np.median(memories)),
    }


def build_question_gen_metric(records):
    if not records:
        return None
    total_chunks = 0
    total_duration_ms = 0
    peak_memory = 0
    for r in records:
        peak_memory        = max(peak_memory, r.get("ram_mb", 0))
        total_duration_ms += r.get("duration_ms", 0)
        total_chunks      += len(r.get("metrics", {}).get("units", []))
    throughput = (
        total_chunks / (total_duration_ms / 60000)
        if total_duration_ms > 0 else 0
    )
    return {"worker": "question_gen", "throughput": throughput, "memory_peak_mb": peak_memory}

def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color":           THEME["text"],
        "axes.labelcolor":      THEME["text_muted"],
        "xtick.color":          THEME["tick"],
        "ytick.color":          THEME["tick"],
        "figure.facecolor":     THEME["page_bg"],
        "axes.facecolor":       THEME["card_bg"],
        "axes.edgecolor":       THEME["card_border"],
    })

def _wrap_text(text: str, width: int = 90) -> str:
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width))


def _style_card_ax(ax: Axes, accent: str) -> None:
    ax.set_facecolor(THEME["card_bg"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(THEME["card_border"])
        spine.set_linewidth(1.0)
    ax.spines["left"].set_color(accent)
    ax.spines["left"].set_linewidth(3.0)


def _style_chart_axes(ax: Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(THEME["card_bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME["card_border"])
    ax.spines["bottom"].set_color(THEME["card_border"])
    if grid_axis == "both":
        ax.grid(alpha=0.35, linestyle="-", color=THEME["grid"], linewidth=0.7)
    elif grid_axis == "y":
        ax.grid(axis="y", alpha=0.35, linestyle="-", color=THEME["grid"], linewidth=0.7)
    else:
        ax.grid(axis="x", alpha=0.35, linestyle="-", color=THEME["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=THEME["tick"], labelsize=7.5, pad=5)


def _panel_title(ax: Axes, title: str, *, pad: float = 14) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold",
                 color=THEME["text"], pad=pad, loc="center")


def _style_card_header(ax: Axes, label: str, subtitle: str, accent: str) -> None:
    ax.set_axis_off()
    ax.set_facecolor(THEME["card_bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    patch = FancyBboxPatch(
        (0.002, 0.02), 0.996, 0.94,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=1.4, edgecolor=THEME["card_border"], facecolor=THEME["card_bg"],
        transform=ax.transAxes, clip_on=False, zorder=1,
    )
    ax.add_patch(patch)

    wash = FancyBboxPatch(
        (0.002, 0.02), 0.32, 0.94,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        linewidth=0, facecolor=accent, alpha=0.10,
        transform=ax.transAxes, clip_on=True, zorder=2,
    )
    ax.add_patch(wash)

    ax.scatter(
        [0.016], [0.48], s=70, color=accent, edgecolor=THEME["card_bg"],
        linewidth=1.3, transform=ax.transAxes, zorder=4, clip_on=False,
    )
    ax.text(
        0.038, 0.68, label, transform=ax.transAxes, fontsize=16.5,
        fontweight="bold", color=THEME["text"], va="center", zorder=4,
    )

    wrapped = _wrap_text(subtitle, width=108)
    n_lines = wrapped.count("\n") + 1
    y_sub   = 0.20 if n_lines == 1 else 0.34
    ax.text(
        0.038, y_sub, wrapped, transform=ax.transAxes,
        fontsize=10.0, color=THEME["text_muted"], va="center", linespacing=1.5, zorder=4,
    )
    ax.plot(
        [0.014, 0.986], [0.04, 0.04], color=THEME["card_border"], linewidth=1.0,
        transform=ax.transAxes, zorder=3,
    )


def plot_worker_overview(ax_tp: Axes, ax_mem: Axes, ax_dur: Axes,
                         metrics: Sequence[dict]) -> None:
    pipeline = list(metrics)
    labels   = [SHORT_WORKER_LABELS.get(m["worker"], worker_style(m["worker"])[0])
                for m in pipeline]
    colors   = [worker_style(m["worker"])[1] for m in pipeline]
    dims     = [worker_style(m["worker"])[2] for m in pipeline]
    x = np.arange(len(pipeline))
    w = 0.29
    # Throughput
    tp_peak = [m["throughput_peak"] for m in pipeline]
    tp_mean = [m["throughput_mean"] for m in pipeline]
    tp_avg  = [m["throughput_avg"] for m in pipeline]

    tp_max = max(tp_peak + tp_mean + tp_avg)

    ax_tp.bar(x - w, tp_peak, width=w, color=colors)
    ax_tp.bar(x,      tp_mean, width=w, color=dims)
    ax_tp.bar(x + w,  tp_avg,  width=w, color=THEME["token_secondary"])
    for i, (p, m, a) in enumerate(zip(tp_peak, tp_mean, tp_avg)):
        ax_tp.text(i - w, p + tp_max*0.02, f"{p:.1f}",
                ha="center", va="bottom", fontsize=6.5,
                color=THEME["text_muted"])

        ax_tp.text(i, m + tp_max*0.02, f"{m:.1f}",
                ha="center", va="bottom", fontsize=6.5,
                color=THEME["text_muted"])

        ax_tp.text(i + w, a + tp_max*0.02, f"{a:.1f}",
                ha="center", va="bottom", fontsize=6.5,
                color=THEME["text_muted"])
    ax_tp.set_xticks(x); ax_tp.set_xticklabels(labels, fontsize=7.5)
    ax_tp.set_ylim(0, tp_max*1.22)
    ax_tp.set_ylabel("req / min", fontsize=8, color=THEME["text_muted"])
    _panel_title(ax_tp, "Throughput", pad=16)
    _style_chart_axes(ax_tp)

    # Memory
    mem_peak = [m["memory_peak_mb"] for m in pipeline]
    mem_mean = [m["memory_mean_mb"] for m in pipeline]
    mem_avg  = [m["memory_avg_mb"] for m in pipeline]

    mem_max = max(mem_peak + mem_mean + mem_avg)

    ax_mem.bar(x - w, mem_peak, width=w, color=colors)
    ax_mem.bar(x,      mem_mean, width=w, color=dims)
    ax_mem.bar(x + w,  mem_avg,  width=w, color=THEME["token_secondary"])
    for i, (p, m, a) in enumerate(zip(mem_peak, mem_mean, mem_avg)):
        ax_mem.text(i - w, p + mem_max*0.02, f"{p:.0f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=THEME["text_muted"])

        ax_mem.text(i, m + mem_max*0.02, f"{m:.0f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=THEME["text_muted"])

        ax_mem.text(i + w, a + mem_max*0.02, f"{a:.0f}",
                ha="center", va="bottom", fontsize=6.5,
                color=THEME["text_muted"])
    ax_mem.set_xticks(x); ax_mem.set_xticklabels(labels, fontsize=7.5)
    ax_mem.set_ylim(0, mem_max*1.22)
    ax_mem.set_ylabel("Memory (MB)", fontsize=8, color=THEME["text_muted"])
    _panel_title(ax_mem, "Memory", pad=16)
    _style_chart_axes(ax_mem)

    # Duration
    dur_peak = [m["duration_peak"] for m in pipeline]
    dur_mean = [m["duration_mean"] for m in pipeline]
    dur_avg  = [m["duration_avg"] for m in pipeline]

    dur_max = max(dur_peak + dur_mean + dur_avg) if dur_peak else 1

    if dur_max >= 1.0:
        dp_peak = dur_peak
        dp_mean = dur_mean
        dp_avg  = dur_avg
        ylabel = "Duration (s)"
        fmt = lambda v: f"{v:.1f}s"
    else:
        dp_peak = [v * 1000 for v in dur_peak]
        dp_mean = [v * 1000 for v in dur_mean]
        dp_avg  = [v * 1000 for v in dur_avg]
        ylabel = "Duration (ms)"
        fmt = lambda v: f"{v:.0f}ms"

    dur_plot_max = max(dp_peak + dp_mean + dp_avg) if dp_peak else 1

    ax_dur.bar(x - w, dp_peak, width=w, color=colors)
    ax_dur.bar(x,      dp_mean, width=w, color=dims)
    ax_dur.bar(x + w,  dp_avg, width=w, color=THEME["token_secondary"])

    for i, (p, m, a) in enumerate(zip(dp_peak, dp_mean, dp_avg)):
        ax_dur.text(i - w, p + dur_plot_max * 0.02, fmt(p),
                    ha="center", va="bottom", fontsize=6.5,
                    color=THEME["text_muted"])

        ax_dur.text(i, m + dur_plot_max * 0.02, fmt(m),
                    ha="center", va="bottom", fontsize=6.5,
                    color=THEME["text_muted"])

        ax_dur.text(i + w, a + dur_plot_max * 0.02, fmt(a),
                    ha="center", va="bottom", fontsize=6.5,
                    color=THEME["text_muted"])

    ax_dur.set_xticks(x)
    ax_dur.set_xticklabels(labels, fontsize=7.5)
    ax_dur.set_ylim(0, dur_plot_max * 1.22)
    ax_dur.set_ylabel(ylabel, fontsize=8, color=THEME["text_muted"])
    _panel_title(ax_dur, "Duration", pad=16)
    _style_chart_axes(ax_dur)
    ax_dur.legend(
    handles=[
        plt.Rectangle((0,0),1,1,  label="Peak"),
        plt.Rectangle((0,0),1,1, label="Mean"),
        plt.Rectangle((0,0),1,1,  label="Median"),
    ],
    fontsize=7,
    frameon=False,
    labelcolor=THEME["text_muted"],
    loc="upper left",
    bbox_to_anchor=(0, 1.08),
    ncol=3,
)


def plot_accuracy_donut(ax: Axes, pairs: Sequence[dict], accent: str,
                        title: str, unit: str, status: str) -> None:
    ax.set_facecolor(THEME["card_bg"])
    ax.set_axis_off()
    if not pairs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color=THEME["text_muted"], transform=ax.transAxes)
        return

    correct, total, acc = match_accuracy(pairs, status)
    _panel_title(ax, title, pad=12)
    incorrect = total - correct

    wedges, _ = ax.pie(
        [max(correct, 0), max(incorrect, 0)], labels=None,
        colors=[accent, THEME["pie_fail"]], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.25, edgecolor=THEME["card_bg"], linewidth=2.0),
        center=(0.5, 0.5), radius=1.6,
    )
    ax.text(0.5, 0.58, f"{acc:.0f}%", ha="center", va="bottom",
            fontsize=17, fontweight="bold", color=THEME["text"])
    ax.text(0.5, 0.28, f"n={total}", ha="center", va="center",
            fontsize=8, color=THEME["text_muted"])
    if unit:
        ax.text(0.5, 0.10, unit, ha="center", va="top",
                fontsize=7.5, color=THEME["text_muted"])
    ax.legend(wedges, [f"Correct ({correct})", f"Wrong ({incorrect})"],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=7.5, frameon=False, handlelength=0.8, columnspacing=1.6,
              labelcolor=THEME["text_muted"], borderpad=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 2.05)
    ax.set_ylim(-1.55, 2.05)


def plot_binary_confusion_matrix(ax: Axes, conf: dict, *, title: str = "") -> None:
    mat = confusion_matrix(conf)
    ax.set_facecolor(THEME["card_bg"])
    if conf["total"] == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color=THEME["text_muted"])
        ax.set_axis_off()
        return
    if title:
        _panel_title(ax, title, pad=16)

    norm = mat.astype(float)
    row_sums = norm.sum(axis=1, keepdims=True)
    np.divide(norm, row_sums, out=norm, where=row_sums > 0)

    cell_rgba = {
        (0,0): THEME["cm_tp"], (0,1): THEME["cm_fn"],
        (1,0): THEME["cm_fp"], (1,1): THEME["cm_tn"],
    }
    display = np.zeros((2, 2, 3))
    for i in range(2):
        for j in range(2):
            base = np.array(to_rgb(cell_rgba[(i, j)]))
            fade = 0.45 + 0.55 * (mat[i, j] / max(mat.max(), 1))
            display[i, j] = base * fade

    ax.imshow(display, aspect="equal", interpolation="nearest",
              extent=(-0.5, 1.5, 1.5, -0.5))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pass (1)", "Fail (0)"], fontsize=7.5)
    ax.set_yticklabels(["Pass (1)", "Fail (0)"], fontsize=7.5)
    ax.set_xlabel("Worker",       fontsize=8, color=THEME["text_muted"], labelpad=14)
    ax.set_ylabel("Ground truth", fontsize=8, color=THEME["text_muted"], labelpad=14)
    ax.tick_params(length=0, pad=7)

    tags = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            count = int(mat[i, j])
            pct   = norm[i, j] * 100 if row_sums[i, 0] > 0 else 0
            ax.text(j, i - 0.18, tags[i][j], ha="center", va="center",
                    fontsize=9, fontweight="bold", color=THEME["text"])
            count_line = f"{count}"
            if row_sums[i, 0] > 0:
                count_line += f"\n{pct:.0f}%"
            ax.text(j, i + 0.16, count_line, ha="center", va="center",
                    fontsize=7, color=THEME["text_muted"])

    ax.text(0.5, -0.38,
            f"Accuracy {conf['accuracy']:.1f}%  ·  n = {conf['total']}",
            transform=ax.transAxes, fontsize=7.5, color=THEME["text_muted"], ha="center")
    ax.set_xlim(-0.60, 1.60)
    ax.set_ylim(1.60, -0.60)

def plot_written_grade_panel(ax_scatter: Axes, ax_error: Axes,
                             comp: dict, accent: str) -> None:
    if not comp["ground_truth"]:
        for ax in (ax_scatter, ax_error):
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=THEME["text_muted"])
        return

    gt  = np.asarray(comp["ground_truth"], dtype=float)
    pr  = np.asarray(comp["predicted"],    dtype=float)
    err = pr - gt
    tol      = 1.0
    warn_tol = 3.0
    lo = float(min(gt.min(), pr.min(), 0))
    hi = float(max(gt.max(), pr.max(), 10))

    _panel_title(ax_scatter, "Written grading quality", pad=16)
    xx = np.linspace(lo, hi, 250)

    ax_scatter.fill_between(xx, xx - warn_tol - 0.25, xx + warn_tol + 0.25,
                            color=THEME["pie_warn"], alpha=0.10, zorder=1)
    ax_scatter.fill_between(xx, xx - tol - 0.25, xx + tol + 0.25,
                            color=THEME["pie_ok"], alpha=0.14, zorder=2)
    ax_scatter.plot([lo, hi], [lo, hi], color=THEME["text_muted"],
                   linestyle="--", linewidth=1.0, alpha=0.8, zorder=3)

    point_colors = [
        THEME["pie_ok"]   if abs(e) <= tol
        else THEME["pie_warn"] if abs(e) <= warn_tol
        else THEME["pie_fail"]
        for e in err
    ]
    ax_scatter.scatter(gt, pr, c=point_colors, s=42, alpha=0.85,
                       edgecolors=THEME["card_bg"], linewidths=0.45, zorder=4)

    stats     = grade_stats(comp)
    stats_box = f"MAE {stats['mae']:.2f}\nRMSE {stats['rmse']:.2f}\nr {stats['r']:.2f}"
    ax_scatter.text(0.97, 0.05, stats_box, transform=ax_scatter.transAxes,
                    fontsize=6.8, color=THEME["text_muted"], va="bottom", ha="right",
                    linespacing=1.4,
                    bbox=dict(boxstyle="round,pad=0.45", facecolor=THEME["card_bg"],
                              edgecolor=THEME["card_border"], alpha=0.95))
    ax_scatter.legend(
        handles=[
            Line2D([0],[0], marker="o", color="w", markerfacecolor=THEME["pie_ok"],   markersize=6, linestyle=""),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=THEME["pie_warn"],  markersize=6, linestyle=""),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=THEME["pie_fail"],  markersize=6, linestyle=""),
        ],
        labels=[f"≤ {tol:.2f}", f"≤ {warn_tol:.1f}", f"> {warn_tol:.1f}"],
        loc="upper left", fontsize=6.5, frameon=False,
        labelcolor=THEME["text_muted"], handletextpad=0.5, borderpad=0.3,
    )
    pad = (hi - lo) * 0.08
    ax_scatter.set_xlim(lo - pad, hi + pad)
    ax_scatter.set_ylim(lo - pad, hi + pad)
    ax_scatter.set_xlabel("Ground truth",    fontsize=8, color=THEME["text_muted"], labelpad=10)
    ax_scatter.set_ylabel("Predicted grade", fontsize=8, color=THEME["text_muted"], labelpad=10)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.tick_params(labelsize=7)
    _style_chart_axes(ax_scatter, grid_axis="both")

    _panel_title(ax_error, "Error distribution", pad=16)
    bins    = np.arange(-10.5, 11.5, 1)
    hist, edges = np.histogram(err, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = [
        THEME["pie_ok"]   if abs(c) <= tol
        else THEME["pie_warn"] if abs(c) <= warn_tol
        else THEME["pie_fail"]
        for c in centers
    ]
    ax_error.bar(centers, hist, width=0.85, color=bar_colors,
                 edgecolor=THEME["card_bg"], linewidth=0.6, alpha=0.9)
    ax_error.axvline(0, color=THEME["text_muted"], linestyle="--",
                     linewidth=1.0, alpha=0.7)

    green_count  = int(np.sum(np.abs(err) <= tol))
    yellow_count = int(np.sum((np.abs(err) > tol) & (np.abs(err) <= warn_tol)))
    red_count    = int(np.sum(np.abs(err) > warn_tol))
    total        = len(err)
    summary = (
        f"right (green)  {green_count} ({100*green_count/total:.1f}%)\n"
        f"warn (yellow) {yellow_count} ({100*yellow_count/total:.1f}%)\n"
        f"wrong (red)    {red_count} ({100*red_count/total:.1f}%)"
    )
    ax_error.text(0.97, 0.95, summary, transform=ax_error.transAxes,
                  ha="right", va="top", fontsize=7, color=THEME["text_muted"],
                  linespacing=1.4,
                  bbox=dict(boxstyle="round,pad=0.45", facecolor=THEME["card_bg"],
                            edgecolor=THEME["card_border"], alpha=0.95))
    ax_error.set_xticks(np.arange(-10, 11, 2))
    ax_error.set_xlabel("Predicted − Ground truth", fontsize=8,
                        color=THEME["text_muted"], labelpad=10)
    ax_error.set_ylabel("Count", fontsize=8, color=THEME["text_muted"], labelpad=10)
    ax_error.tick_params(labelsize=7)
    _style_chart_axes(ax_error)


def plot_llm_usage_panel(ax_pie, ax_kpi, ax_chunks, usage, worker_key):
    ax_pie.clear(); ax_kpi.clear(); ax_chunks.clear()

    totals    = usage_totals(usage)
    in_tok    = totals["tokens_input"]
    out_tok   = totals["tokens_output"]
    total_tok = totals["tokens_total"]

    units         = usage.get("units", [])
    n             = len(units)
    input_by      = np.array([u.get("tokens_input",  0) for u in units], float)
    output_by     = np.array([u.get("tokens_output", 0) for u in units], float)
    api           = np.array([u.get("api_calls",      0) for u in units], float)
    unit_x_labels = [u.get("label") or str(u.get("unit_index", i+1)) for i, u in enumerate(units)]
    unit_name     = usage.get("unit_label") or "unit"

    DONUT_INPUT_COLOR  = "#58A6FF"
    DONUT_OUTPUT_COLOR = "#D29922"

    ax_pie.set_facecolor(THEME["card_bg"])
    ax_pie.set_axis_off()
    if total_tok <= 0:
        ax_pie.text(0.5, 0.5, "No usage", ha="center", va="center",
                    color=THEME["text_muted"], transform=ax_pie.transAxes)
    else:
        ax_pie.pie(
            [in_tok, out_tok],
            colors=[DONUT_INPUT_COLOR, DONUT_OUTPUT_COLOR],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.18, edgecolor=THEME["card_bg"]),
            radius=1.3,
        )
        ax_pie.text(0, 0.08,  f"{total_tok:,}", ha="center",
                    fontsize=16, fontweight="bold", color=THEME["text"])
        ax_pie.text(0, -0.28, "total tokens", ha="center",
                    fontsize=8, color=THEME["text_muted"])
        ax_pie.legend(
            [Line2D([0],[0], color=DONUT_INPUT_COLOR,  lw=6),
             Line2D([0],[0], color=DONUT_OUTPUT_COLOR, lw=6)],
            ["Input tokens", "Output tokens"],
            loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2,
            frameon=False, fontsize=7, labelcolor=THEME["text_muted"],
            handlelength=1.5, columnspacing=1.0,
        )

    ax_kpi.set_axis_off()
    if n == 0:
        ax_kpi.text(0.5, 0.5, "No usage data", ha="center", va="center",
                    color=THEME["text_muted"], fontsize=8, transform=ax_kpi.transAxes)
        ax_chunks.set_axis_off()
        ax_chunks.set_facecolor(THEME["card_bg"])
        ax_chunks.text(0.5, 0.5, "No usage data", ha="center", va="center",
                       color=THEME["text_muted"], fontsize=8, transform=ax_chunks.transAxes)
        return

    peak_in  = input_by.max()
    peak_out = output_by.max()

    kpis = [
        ("INPUT",      in_tok,         "#58A6FF"),
        ("OUTPUT",     out_tok,        "#D29922"),
        ("API CALLS",  int(api.sum()), "#3FB950"),
        ("PEAK INPUT",  int(peak_in),  "#58A6FF"),
        ("PEAK OUTPUT", int(peak_out), "#D29922"),
    ]

    n_cols   = 3
    n_rows   = math.ceil(len(kpis) / n_cols)
    margin_x = 0.015
    margin_y = 0.04
    gap_x    = 0.025
    gap_y    = 0.10
    card_w   = (1 - 2*margin_x - gap_x*(n_cols-1)) / n_cols
    card_h   = (1 - 2*margin_y - gap_y*(n_rows-1)) / n_rows
    xs       = [margin_x + c*(card_w + gap_x) for c in range(n_cols)]
    # row 0 sits at the top (y closest to 1); FancyBboxPatch anchors bottom-left
    ys       = [1 - margin_y - (r+1)*card_h - r*gap_y for r in range(n_rows)]

    fig      = ax_kpi.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv      = ax_kpi.transAxes.inverted()

    def text_width_axes_frac(s, fontsize, weight="normal"):
        t = ax_kpi.text(0, 0, s, fontsize=fontsize, fontweight=weight, alpha=0)
        bbox = t.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform(bbox)
        t.remove()
        return x1 - x0

    stripe_w   = 0.014
    pad_frac   = stripe_w + 0.016
    right_pad  = 0.012
    max_text_w = card_w - pad_frac - right_pad

    for i, (label, val, col) in enumerate(kpis):
        row, c = divmod(i, n_cols)
        x, y   = xs[c], ys[row]
        val_str = f"{val:,}"
        val_fs  = 12
        while val_fs > 6.5 and text_width_axes_frac(val_str, val_fs, "bold") > max_text_w:
            val_fs -= 0.5
        lbl_fs = 7
        while lbl_fs > 4.5 and text_width_axes_frac(label, lbl_fs) > max_text_w * 0.92:
            lbl_fs -= 0.5

        ax_kpi.add_patch(FancyBboxPatch(
            (x, y), card_w, card_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=THEME["card_bg"], edgecolor=THEME["card_border"],
            linewidth=1, transform=ax_kpi.transAxes,
        ))
        ax_kpi.add_patch(FancyBboxPatch(
            (x, y), stripe_w, card_h,
            boxstyle="round,pad=0",
            facecolor=col, linewidth=0, transform=ax_kpi.transAxes,
        ))
        text_x = x + pad_frac
        ax_kpi.text(text_x, y + card_h*0.66, label,   fontsize=lbl_fs,
                    color=THEME["text_muted"], transform=ax_kpi.transAxes)
        ax_kpi.text(text_x, y + card_h*0.30, val_str, fontsize=val_fs, fontweight="bold",
                    color=THEME["text"],       transform=ax_kpi.transAxes)

    ax_chunks.set_facecolor(THEME["card_bg"])
    COLOR_IN  = "#58A6FF"
    COLOR_OUT = "#D29922"
    COLOR_API = "#F0883E"

    x = np.arange(n)
    ax_chunks.fill_between(x, input_by,  alpha=0.15, color=COLOR_IN)
    line_in, = ax_chunks.plot(x, input_by,  linewidth=2.0, color=COLOR_IN,
                               label="Input tokens")
    ax_chunks.fill_between(x, output_by, alpha=0.15, color=COLOR_OUT)
    line_out, = ax_chunks.plot(x, output_by, linewidth=2.0, color=COLOR_OUT,
                                label="Output tokens")

    peak_in_idx  = int(np.argmax(input_by))
    peak_out_idx = int(np.argmax(output_by))
    ax_chunks.scatter([peak_in_idx],  [input_by[peak_in_idx]],
                      color=COLOR_IN,  s=55, zorder=5)
    ax_chunks.scatter([peak_out_idx], [output_by[peak_out_idx]],
                      color=COLOR_OUT, s=55, zorder=5)

    ax_api = ax_chunks.twinx()
    ax_api.set_facecolor("none")
    ax_api.spines["right"].set_color(COLOR_API)
    ax_api.spines["right"].set_linewidth(1.2)
    ax_api.tick_params(axis="y", colors=COLOR_API, labelsize=7)
    ax_api.yaxis.label.set_color(COLOR_API)
    line_api, = ax_api.plot(
        x, api, color="green", linewidth=1.6, linestyle="--",
        marker="o", markersize=4.5, markerfacecolor=COLOR_API,
        markeredgecolor=THEME["card_bg"], markeredgewidth=0.8,
        label="API calls", zorder=6,
    )
    ax_api.set_ylabel("API calls", fontsize=8, color=COLOR_API, labelpad=8)
    api_max = int(api.max()) if api.max() > 0 else 1
    ax_api.set_ylim(-0.5, api_max * 2.2)
    ax_api.set_yticks(range(0, api_max + 1))
    for sp in ("top", "left", "bottom"):
        ax_api.spines[sp].set_visible(False)

    max_ticks = 15
    if n <= max_ticks:
        tick_idx = list(range(n))
    else:
        step = math.ceil(n / max_ticks)
        tick_idx = list(range(0, n, step))
        if tick_idx[-1] != n - 1:
            tick_idx.append(n - 1)
    ax_chunks.set_xticks(tick_idx)
    ax_chunks.set_xticklabels([unit_x_labels[i] for i in tick_idx])
    ax_chunks.set_title(f"Token usage per {unit_name}", fontsize=10,
                         fontweight="bold", color=THEME["text"], loc="left")
    ax_chunks.set_ylabel("tokens", fontsize=8, color=THEME["text_muted"])
    ax_chunks.tick_params(colors=THEME["text_muted"], labelsize=7)
    for spine in ax_chunks.spines.values():
        spine.set_color(THEME["card_border"])
    ax_chunks.legend(
        handles=[line_in, line_out, line_api],
        labels=["Input tokens", "Output tokens", "API calls"],
        loc="lower left", bbox_to_anchor=(0.0, -0.32), ncol=3,
        frameon=False, fontsize=6.5, labelcolor=THEME["text_muted"],
        handlelength=1.5, columnspacing=1.0, borderaxespad=0,
    )


def _render_llm_card(
    fig: Figure, gs: gridspec.SubplotSpec,
    usage: dict, worker_key: str, metrics: Sequence[dict],
    subtitle: str,
) -> list[Axes]:
    label, accent, _dim = worker_style(worker_key)
    ax_card = fig.add_subplot(gs)
    _style_card_ax(ax_card, accent)
    ax_card.set_axis_off()

    card = gs.subgridspec(2, 1, height_ratios=[0.16, 1], hspace=0.28)
    ax_hdr = fig.add_subplot(card[0])
    _style_card_header(ax_hdr, label, subtitle, accent)

    body = card[1].subgridspec(
        2, 2, height_ratios=[0.72, 1.28],
        width_ratios=[0.86, 1.0], hspace=0.42, wspace=0.26,
    )
    ax_pie    = fig.add_subplot(body[0, 0])
    ax_txt    = fig.add_subplot(body[0, 1])
    ax_chunks = fig.add_subplot(body[1, :])
    ax_pie.set_facecolor(THEME["card_bg"])
    ax_txt.set_facecolor(THEME["card_bg"])
    ax_chunks.set_facecolor(THEME["card_bg"])
    plot_llm_usage_panel(ax_pie, ax_txt, ax_chunks, usage, worker_key)
    return [ax_card, ax_hdr, ax_pie, ax_txt, ax_chunks]


def _save_individual_cards(
    fig: Figure, card_axes: dict[str, list[Axes]], output_path: Path,
) -> list[Path]:
    """Save each named group of axes as its own standalone image, alongside
    the combined dashboard PNG. Files land next to output_path, named
    '<output_stem>__<card_name>.png'."""
    saved: list[Path] = []
    fig.canvas.draw()  # ensure all artists have up-to-date extents
    for name, axes_list in card_axes.items():
        axes_list = [a for a in axes_list if a is not None]
        if not axes_list:
            continue
        renderer = fig.canvas.get_renderer()
        bboxes = [a.get_tightbbox(renderer) for a in axes_list]
        bboxes = [b for b in bboxes if b is not None]
        if not bboxes:
            continue
        full_bbox = Bbox.union(bboxes)
        # pad slightly so card borders/strokes aren't clipped
        full_bbox = full_bbox.padded(6)
        card_path = output_path.with_name(f"{output_path.stem}__{name}{output_path.suffix}")
        fig.savefig(
            card_path, dpi=175, bbox_inches=full_bbox.transformed(fig.dpi_scale_trans.inverted()),
            facecolor=THEME["page_bg"], pad_inches=0.15,
        )
        saved.append(card_path)
    return saved


def build_report_image(data: dict, output_path: Path,
                       title: str = REPORT_TITLE) -> Path:
    _apply_rc()

    # ── Pull all sections ────────────────────────────────────────────────────
    detection_worker   = data["detection_worker"]
    complete_worker    = data["complete_worker"]
    written_worker     = data["written_worker"]
    Grading_worker     = data["Grading_worker"]
    question_generation = data["question_generation"]
    recognition        = data["recognition"]   # {"mcq": [...], "choices": [...]}
    grading            = data["grading"]        # {"mcq": {...}, "choices": {...}, "essays": {...}}

    # ── Pre-compute grading stats for header subtitles ───────────────────────
    # MCQ grading: binary pass/fail lists → confusion
    mcq_conf   = binary_confusion(grading["mcq"]["ground_truth"],
                                  grading["mcq"]["predicted"])
    # Choices (completion) grading: binary lists → confusion
    choices_conf = binary_confusion(grading["choices"]["ground_truth"],
                                    grading["choices"]["predicted"])
    # Essays: continuous scores
    essay_stats = grade_stats(grading["essays"])

    # ── LLM usage ────────────────────────────────────────────────────────────

    # written_worker token log → per-question units for the extraction card
    units_written_ext = [
        {**i["tokens"], "unit_index": idx + 1, "label": f"q{idx+1}"}
        for idx, i in enumerate(written_worker)
        if "tokens" in i
    ]
    usage_written_extraction = {"unit_label": "question", "units": units_written_ext}

    # grading usage for written answers — each Grading_worker entry's
    # tokens.usage_written.units is a per-answer list for that grading run;
    # flatten across all runs, tagging units with their run index to keep them distinct
    units_written_grading: list[dict] = []
    for run_idx, entry in enumerate(Grading_worker):
        usage_written = entry.get("tokens", {}).get("usage_written")
        if not usage_written:
            continue
        for u in usage_written.get("units", []):
            units_written_grading.append({
                **u,
                "unit_index": len(units_written_grading) + 1,
                "label": f"{u.get('label', u.get('unit_index'))[:3]} r{run_idx + 1}",
            })
    usage_written_grading = {"unit_label": "answer", "units": units_written_grading}

    # grading usage for the "complete" (fill-in-the-blank) sheet — each
    # Grading_worker entry's tokens.usage_complete.unit is a single dict per run
    units_choices_grading: list[dict] = []
    for run_idx, entry in enumerate(Grading_worker):
        usage_complete = entry.get("tokens", {}).get("usage_complete")
        if not usage_complete:
            continue
        unit = usage_complete.get("unit")
        if not unit:
            continue
        units_choices_grading.append({
            **unit,
            "unit_index": run_idx + 1,
            "label": unit.get("label", f"r{run_idx + 1}"),
        })
    usage_choices_grading = {"unit_label": "sheet", "units": units_choices_grading}

    # question generation
    usage_question_gen = build_question_gen_usage(question_generation)

    # ── Worker perf metrics ───────────────────────────────────────────────────
    worker_metrics = [
        build_worker_metrics("detection_worker",   detection_worker),
        build_worker_metrics("complete_worker",    complete_worker),
        build_worker_metrics("written_worker",     written_worker),
        build_worker_metrics("Grading_worker",     Grading_worker),
        build_worker_metrics("question_generation", question_generation),
    ]

    # ── Recognition accuracy for subtitle ────────────────────────────────────
    mcq_acc     = match_accuracy(recognition["mcq"],     "MCQ")[2]
    choices_acc = match_accuracy(recognition["choices"], "complete")[2]

    # ── Figure layout ────────────────────────────────────────────────────────
    #   Row 0 (0.13) : overview (throughput / memory / duration)
    #   Row 1 (0.20) : recognition donuts  ← NEW
    #   Row 2 (0.28) : grading panels      ← NEW
    #   Row 3 (0.39) : LLM usage cards
    fig = plt.figure(figsize=(17, 27), facecolor=THEME["page_bg"])
    fig.text(0.5, 0.997, title, ha="center", va="top", fontsize=22,
             fontweight="bold", color=THEME["text"])
    fig.text(
        0.5, 0.975,
        "Sheet detection · Complete · Written · Grading   |   Question Generation",
        ha="center", va="top", fontsize=8.5, color=THEME["text_muted"],
    )

    # Tracks {card_name: [axes...]} so each card can also be saved as its own image
    card_axes: dict[str, list[Axes]] = {}

    root = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[0.13, 0.20, 0.28, 0.39],
        left=0.06, right=0.965, top=0.935, bottom=0.030, hspace=0.52,
    )

    # ── Row 0: Worker overview ────────────────────────────────────────────────
    ov_wrap = root[0].subgridspec(1, 1)
    ax_ov_card = fig.add_subplot(ov_wrap[0])
    _style_card_ax(ax_ov_card, THEME["text_muted"])
    ax_ov_card.set_axis_off()
    overview = ov_wrap[0].subgridspec(1, 3, wspace=0.38)
    ax_dur = fig.add_subplot(overview[0])
    ax_tp  = fig.add_subplot(overview[1])
    ax_mem = fig.add_subplot(overview[2])
    for ax in (ax_tp, ax_mem, ax_dur):
        ax.set_facecolor(THEME["card_bg"])
    plot_worker_overview(ax_tp, ax_mem, ax_dur, worker_metrics)
    card_axes["overview"] = [ax_ov_card, ax_dur, ax_tp, ax_mem]

    # ── Row 1: Recognition ───────────────────────────────────────────────────
    rec_label, rec_color, _ = worker_style("recognition")
    rec_wrap  = root[1].subgridspec(1, 1)
    ax_r_card = fig.add_subplot(rec_wrap[0])
    _style_card_ax(ax_r_card, rec_color)
    ax_r_card.set_axis_off()

    rec_outer = rec_wrap[0].subgridspec(2, 1, height_ratios=[0.35, 1], hspace=0.35)
    ax_r_hdr  = fig.add_subplot(rec_outer[0])
    _style_card_header(
        ax_r_hdr, rec_label,
        f"MCQ {mcq_acc:.0f}%  ·  Completion {choices_acc:.0f}%",
        rec_color,
    )

    # Two donuts: MCQ bubbles + Completion fill-in
    rec_pies = rec_outer[1].subgridspec(1, 2, wspace=0.48)
    ax_rm    = fig.add_subplot(rec_pies[0])   # MCQ
    ax_rc    = fig.add_subplot(rec_pies[1])   # Choices / completion

    plot_accuracy_donut(
        ax_rm, recognition["mcq"], rec_color,
        "MCQ", "per answer", "MCQ",
    )
    plot_accuracy_donut(
        ax_rc, recognition["choices"], THEME["token_secondary"],
        "Completion", "per character", "complete",
    )
    card_axes["recognition"] = [ax_r_card, ax_r_hdr, ax_rm, ax_rc]

    # ── Row 2: Grading ───────────────────────────────────────────────────────
    grd_label, grd_color, _ = worker_style("grading")
    grd_wrap  = root[2].subgridspec(1, 1)
    ax_g_card = fig.add_subplot(grd_wrap[0])
    _style_card_ax(ax_g_card, grd_color)
    ax_g_card.set_axis_off()

    grd_outer = grd_wrap[0].subgridspec(2, 1, height_ratios=[0.28, 1], hspace=0.35)
    ax_g_hdr  = fig.add_subplot(grd_outer[0])
    _style_card_header(
        ax_g_hdr, grd_label,
        (f"MCQ {mcq_conf['accuracy']:.0f}%  ·  "
         f"Choices {choices_conf['accuracy']:.0f}%  ·  "
         f"Essays MAE {essay_stats['mae']:.2f}  ·  r {essay_stats['r']:.2f}"),
        grd_color,
    )

    # Body: [MCQ confusion | Choices confusion | Essay scatter + error bar]
    grd_body  = grd_outer[1].subgridspec(1, 3, width_ratios=[1, 1, 2.2], wspace=0.44)

    ax_mcq_cm     = fig.add_subplot(grd_body[0])
    ax_choices_cm = fig.add_subplot(grd_body[1])
    essay_gs      = grd_body[2].subgridspec(1, 2, width_ratios=[1, 0.75], wspace=0.44)
    ax_essay_scatter = fig.add_subplot(essay_gs[0])
    ax_essay_err     = fig.add_subplot(essay_gs[1])

    for ax in (ax_mcq_cm, ax_choices_cm, ax_essay_scatter, ax_essay_err):
        ax.set_facecolor(THEME["card_bg"])

    plot_binary_confusion_matrix(ax_mcq_cm,     mcq_conf,     title="MCQ grading")
    plot_binary_confusion_matrix(ax_choices_cm, choices_conf, title="Choices grading")
    plot_written_grade_panel(ax_essay_scatter, ax_essay_err,
                             grading["essays"], grd_color)
    card_axes["grading"] = [ax_g_card, ax_g_hdr, ax_mcq_cm, ax_choices_cm,
                             ax_essay_scatter, ax_essay_err]

    # ── Row 3: LLM usage cards ────────────────────────────────────────────────
    llm_row = root[3].subgridspec(1, 4, wspace=0.22)
    card_axes["text_extraction"] = _render_llm_card(
        fig, llm_row[0], usage_written_extraction, "written_extraction", worker_metrics,
        "written text extraction | tokens & API",
    )
    card_axes["written_grading"] = _render_llm_card(
        fig, llm_row[1], usage_written_grading, "written_grading", worker_metrics,
        "questions + answers evaluation | tokens & API",
    )
    card_axes["choices_grading"] = _render_llm_card(
        fig, llm_row[2], usage_choices_grading, "choices_grading", worker_metrics,
        "fill-in-the-blank sheet evaluation | tokens & API",
    )
    card_axes["question_generation"] = _render_llm_card(
        fig, llm_row[3], usage_question_gen, "question_gen", worker_metrics,
        "separate worker · tokens & API",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=175, bbox_inches="tight",
                facecolor=THEME["page_bg"], pad_inches=0.50)

    # Also save every individual card/panel as its own standalone image
    individual_paths = _save_individual_cards(fig, card_axes, output_path)
    for p in individual_paths:
        print(f"  card saved: {p.resolve()}")

    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the BLASE observability report from a JSON data file.")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("report_output/async_workers_report.png"))
    parser.add_argument("--data-json", default="./matrics/data.json", type=Path)
    parser.add_argument("--title", type=str, default=REPORT_TITLE)
    args = parser.parse_args()

    data = load_report_data(args.data_json)
    out  = build_report_image(data, args.output, title=args.title)
    print(f"Report saved: {out.resolve()}")


if __name__ == "__main__":
    main()