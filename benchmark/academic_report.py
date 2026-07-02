from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


TOTAL_FAIL_PCT = 24.0
REPORT_TITLE = "Blase Academic Report"

_TOTAL_BAND_PCT_BREAKS = (TOTAL_FAIL_PCT, 40.0, 60.0, 80.0, 100.0)


# ---------------------------------------------------------------------------
# Max-score inference  (replaces ExamConfig)
# ---------------------------------------------------------------------------

def infer_max_score(series: pd.Series) -> float:
    """Round up to the nearest 'standard' ceiling, or use the observed max."""
    max_val = float(series.max())
    if max_val <= 0:
        return 1.0
    for standard in [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 150, 200, 500, 1000]:
        if max_val <= standard:
            return float(standard)
    return max_val


def derive_max_scores(df: pd.DataFrame) -> dict[str, float]:
    """
    Return {'mcq', 'complete', 'written', 'total'} maximums from the DataFrame.

    Priority (highest → lowest):
      1. Explicit  *_total_score / max_* columns already in the df
         (e.g. choice_total_score, max_mcq – whatever the caller aliased them to)
      2. Rounding the observed score ceiling to a standard value
    """
    def _col_max(candidates: list[str]) -> float | None:
        for c in candidates:
            if c in df.columns and df[c].max() > 0:
                return float(df[c].max())
        return None

    max_mcq      = _col_max(["max_mcq",      "choice_total_score"]) or infer_max_score(df["mcq"])
    max_complete = _col_max(["max_complete",  "complete_total_score"]) or infer_max_score(df["complete"])
    max_written  = _col_max(["max_written",   "written_total_score"]) or infer_max_score(df["written"])

    # Total: prefer an explicit column, then sum of the three section maxes
    max_total = _col_max(["max_total", "total_total_score"])
    if max_total is None:
        max_total = max_mcq + max_complete + max_written

    return {
        "mcq":      max_mcq,
        "complete": max_complete,
        "written":  max_written,
        "total":    max_total,
    }


# ---------------------------------------------------------------------------
# ScoreType  (now receives its max from derive_max_scores, not ExamConfig)
# ---------------------------------------------------------------------------

class ScoreType:
    def __init__(
        self,
        key: str,
        label: str,
        column: str | None,
        max_score: float,
        color: str,
        dim: str,
        pie_colors: tuple[str, ...],
    ) -> None:
        self.key        = key
        self.label      = label
        self.column     = column
        self.max_score  = max_score
        self.color      = color
        self.dim        = dim
        self.pie_colors = pie_colors

    def raw_series(self, df: pd.DataFrame) -> pd.Series:
        return df["total"] if self.column is None else df[self.column]

    def pct_series(self, df: pd.DataFrame) -> pd.Series:
        return (self.raw_series(df) / self.max_score) * 100


def build_score_types(maxes: dict[str, float]) -> list[ScoreType]:
    """Build the four ScoreType objects from a max-score dict."""
    return [
        ScoreType("mcq", "MCQ", "mcq", maxes["mcq"],
                  "#58A6FF", "#1F3A5F",
                  ("#1F3A5F", "#264A73", "#3D7DD6", "#58A6FF", "#79C0FF")),
        ScoreType("complete", "Complete", "complete", maxes["complete"],
                  "#F0883E", "#4A2C14",
                  ("#4A2C14", "#6B3F1E", "#B86B28", "#F0883E", "#FBB06A")),
        ScoreType("written", "Written", "written", maxes["written"],
                  "#BC8CFF", "#3B2A5C",
                  ("#3B2A5C", "#5A3F8A", "#8B5FCF", "#BC8CFF", "#D2B4FF")),
        ScoreType("total", "Total", None, maxes["total"],
                  "#3FB950", "#1A3D24",
                  ("#F85149", "#DB6D28", "#D29922", "#3FB950", "#58A6FF")),
    ]


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

class DarkTheme:
    def __init__(self) -> None:
        self.page_bg          = "#0D1117"
        self.card_bg          = "#161B22"
        self.card_border      = "#30363D"
        self.chart_bg         = "#0D1117"
        self.text             = "#E6EDF3"
        self.text_muted       = "#8B949E"
        self.grid             = "#21262D"
        self.perf_track       = "#21262D"
        self.tick             = "#6E7681"
        self.total_pie_colors = (
            "#F85149", "#DB6D28", "#D29922", "#3FB950", "#58A6FF",
        )


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def enrich_dataframe(df: pd.DataFrame, maxes: dict[str, float]) -> pd.DataFrame:
    """Add 'total' and 'pct' columns if absent."""
    out = df.copy()
    if "total" not in out.columns:
        out["total"] = out["mcq"] + out["complete"] + out["written"]
    if "pct" not in out.columns:
        out["pct"] = (out["total"] / maxes["total"]) * 100
    return out


def load_scores_json(data) -> pd.DataFrame:
    records = (
        data.get("students", data.get("data", data))
        if isinstance(data, dict) else data
    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Band helpers
# ---------------------------------------------------------------------------

def section_score_bands(max_score: float, n_bands: int = 5) -> list[tuple[str, float, float]]:
    edges = np.linspace(0, max_score, n_bands + 1)
    bands: list[tuple[str, float, float]] = []
    for i in range(n_bands):
        lo, hi = float(edges[i]), float(edges[i + 1])
        hi_assign = hi if i < n_bands - 1 else max_score + 0.001
        bands.append((f"{int(round(lo))}–{int(round(hi))}", lo, hi_assign))
    return bands


def total_score_bands(max_score: float) -> list[tuple[str, float, float]]:
    edges = [max_score * p / 100 for p in (0.0, *_TOTAL_BAND_PCT_BREAKS)]
    n = len(edges) - 1
    bands: list[tuple[str, float, float]] = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        hi_assign = hi if i < n - 1 else max_score + 0.001
        label = f"Fail (<{hi:.0f})" if i == 0 else f"{lo:.0f}–{hi:.0f}"
        bands.append((label, lo, hi_assign))
    return bands


def assign_band(value: float, bands: Sequence[tuple[str, float, float]]) -> str:
    for label, lo, hi in bands:
        if lo <= value <= hi:
            return label
    return bands[-1][0]


def band_color(
    value: float,
    bands: Sequence[tuple[str, float, float]],
    colors: Sequence[str],
) -> str:
    for (_, lo, hi), color in zip(bands, colors):
        if lo <= value <= hi:
            return color
    return colors[-1]


def range_distribution(
    values: pd.Series,
    bands: Sequence[tuple[str, float, float]],
    labels: Sequence[str],
) -> pd.Series:
    assigned = values.apply(lambda v: assign_band(v, bands))
    return assigned.value_counts().reindex(labels, fill_value=0)


def section_stats_raw(raw: pd.Series, max_score: float) -> dict[str, float]:
    return {
        "avg":     raw.mean(),
        "median":  raw.median(),
        "std":     raw.std(),
        "avg_pct": (raw.mean() / max_score) * 100,
    }


# ---------------------------------------------------------------------------
# Bin helpers
# ---------------------------------------------------------------------------

def hist_bins_aligned_to_bands(
    max_score: float,
    bands: Sequence[tuple[str, float, float]],
    n_bins: int = 10,
) -> np.ndarray:
    base = np.linspace(0, max_score, n_bins + 1)
    raw_band_edges = {0.0, max_score}
    for _, lo, hi in bands:
        raw_band_edges.add(lo)
        raw_band_edges.add(hi if hi <= max_score else max_score)

    merged  = sorted(set(np.round(base, 6)) | set(np.round(list(raw_band_edges), 6)))
    min_gap = max(max_score * 0.01, 1e-6)
    cleaned = [merged[0]]
    for edge in merged[1:]:
        if edge - cleaned[-1] >= min_gap:
            cleaned.append(edge)
        else:
            cleaned[-1] = edge
    cleaned[-1] = max_score
    return np.array(cleaned)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _apply_rc(theme: DarkTheme) -> None:
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color":        theme.text,
        "axes.labelcolor":   theme.text_muted,
        "xtick.color":       theme.tick,
        "ytick.color":       theme.tick,
        "figure.facecolor":  theme.page_bg,
        "axes.facecolor":    theme.chart_bg,
        "axes.edgecolor":    theme.card_border,
    })


def _style_chart_axes(ax: Axes, theme: DarkTheme) -> None:
    ax.set_facecolor(theme.chart_bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme.card_border)
    ax.spines["bottom"].set_color(theme.card_border)
    ax.grid(axis="y", alpha=0.55, linestyle="-", color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme.tick, labelsize=7.5)


def _style_card_ax(ax: Axes, theme: DarkTheme, accent: str) -> None:
    ax.set_facecolor(theme.card_bg)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(theme.card_border)
        spine.set_linewidth(1.0)
    ax.spines["left"].set_color(accent)
    ax.spines["left"].set_linewidth(3.0)


def plot_histogram(
    ax: Axes,
    raw: pd.Series,
    st: ScoreType,
    theme: DarkTheme,
    bands: Sequence[tuple[str, float, float]],
    colors: Sequence[str],
) -> None:
    bin_edges = hist_bins_aligned_to_bands(st.max_score, bands)
    counts, _, patches = ax.hist(
        raw, bins=bin_edges, edgecolor=theme.card_bg,
        linewidth=0.9, alpha=0.9, zorder=2,
    )
    for i, patch in enumerate(patches):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        patch.set_facecolor(band_color(bin_center, bands, colors))

    ymax = max(counts) if len(counts) else 1
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                count + ymax * 0.04,
                str(int(count)), ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=theme.text,
            )

    ax.set_xlim(0, st.max_score)
    ax.set_xticks(bin_edges)
    tick_labels = [f"{e:.0f}" if e == int(e) else f"{e:.1f}" for e in bin_edges]
    ax.set_xticklabels(tick_labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_xlabel(f"Score (0 – {st.max_score:.0f})", fontsize=8, color=theme.text_muted)
    ax.set_ylabel("Students", fontsize=8, color=theme.text_muted)
    _style_chart_axes(ax, theme)


def plot_range_pie(
    ax: Axes,
    values: pd.Series,
    bands: Sequence[tuple[str, float, float]],
    band_labels: Sequence[str],
    colors: Sequence[str],
    theme: DarkTheme,
) -> None:
    counts  = range_distribution(values, bands, band_labels)
    total_n = counts.sum()

    wedges, _, autotexts = ax.pie(
        counts.values.astype(float),
        labels=None, colors=list(colors),
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.5, edgecolor=theme.card_bg, linewidth=1.5),
        autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
        pctdistance=0.75,
        textprops=dict(fontsize=7.5, color=theme.text, fontweight="bold"),
        radius=1.2,
    )
    legend_labels = [
        f"{lbl}  {int(c)} ({c / total_n * 100:.0f}%)" if total_n else f"{lbl}  0"
        for lbl, c in zip(band_labels, counts.values)
    ]
    ax.legend(
        wedges, legend_labels,
        loc="center left", bbox_to_anchor=(1.08, 0.5),
        fontsize=6.5, frameon=False, handlelength=0.7,
        labelcolor=theme.text_muted,
    )
    ax.set_facecolor(theme.chart_bg)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")


def plot_section_performance(
    ax: Axes,
    df: pd.DataFrame,
    types: Sequence[ScoreType],
    theme: DarkTheme,
) -> None:
    section_types = [t for t in types if t.key != "total"]
    y_pos = np.arange(len(section_types))
    xmax  = max(st.max_score for st in section_types) * 1.15

    for i, st in enumerate(section_types):
        raw      = st.raw_series(df)
        avg, med = raw.mean(), raw.median()
        y        = y_pos[i]

        ax.barh(y, xmax, height=0.62, color=st.dim, edgecolor="none", alpha=0.55, zorder=0)
        ax.barh(y, st.max_score, height=0.4, color=theme.perf_track,
                edgecolor=theme.card_border, linewidth=0.5, zorder=1)
        ax.barh(y, avg, height=0.4, color=st.color, alpha=0.95, edgecolor="none", zorder=2)
        ax.plot(med, y, marker="D", markersize=9, color=theme.text,
                markeredgecolor=st.color, markeredgewidth=1.8, zorder=4)

        ax.text(-0.5, y, st.label, va="center", ha="right",
                fontsize=10, fontweight="bold", color=st.color)
        ax.text(avg * 0.42, y, f"{avg:.1f}", va="center", ha="center",
                fontsize=8.5, fontweight="bold", color=theme.page_bg, zorder=5)
        ax.text(med, y + 0.32, f"{med:.1f}", va="bottom", ha="center",
                fontsize=7.5, color=theme.text_muted)
        ax.text(st.max_score + 0.3, y, f"/ {st.max_score:.0f}",
                va="center", ha="left", fontsize=7.5, color=theme.text_muted)

    ax.set_xlim(-2.2, xmax)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_facecolor(theme.card_bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(theme.card_border)
    ax.grid(axis="x", alpha=0.55, color=theme.grid, linewidth=0.8)
    ax.set_xlabel("Points", fontsize=8, color=theme.text_muted)
    ax.tick_params(colors=theme.tick)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], color=theme.perf_track, lw=8, label="Max"),
        Line2D([0], [0], color="#58A6FF",         lw=8, label="Avg"),
        Line2D([0], [0], marker="D", color=theme.text, lw=0, markersize=8, label="Median"),
    ]
    leg = ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5,
                    facecolor=theme.chart_bg, edgecolor=theme.card_border,
                    labelcolor=theme.text_muted)
    leg.get_frame().set_alpha(0.9)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class TypePanel:
    def __init__(self, score_type: ScoreType) -> None:
        self.score_type = score_type

    def render(
        self,
        fig: Figure,
        outer_gs: gridspec.SubplotSpec,
        df: pd.DataFrame,
        theme: DarkTheme,
    ) -> None:
        st    = self.score_type
        raw   = st.raw_series(df)
        stats = section_stats_raw(raw, st.max_score)

        ax_card = fig.add_subplot(outer_gs)
        _style_card_ax(ax_card, theme, st.color)
        ax_card.set_axis_off()

        inner = outer_gs.subgridspec(
            2, 2, height_ratios=[0.16, 1],
            width_ratios=[0.9, 1.1], hspace=0.22, wspace=0.42,
        )

        ax_title = fig.add_subplot(inner[0, :])
        ax_title.set_axis_off()
        ax_title.set_facecolor(theme.card_bg)
        ax_title.text(
            0.02, 0.55, st.label,
            transform=ax_title.transAxes,
            fontsize=12, fontweight="bold", color=st.color, va="center",
        )
        ax_title.text(
            0.22, 0.55,
            f"avg {stats['avg']:.1f}/{st.max_score:.0f}  ·  "
            f"{stats['avg_pct']:.0f}%  ·  med {stats['median']:.1f}  ·  σ {stats['std']:.1f}",
            transform=ax_title.transAxes,
            fontsize=8.5, color=theme.text_muted, va="center",
        )

        ax_hist = fig.add_subplot(inner[1, 0])
        ax_pie  = fig.add_subplot(inner[1, 1])

        if st.key == "total":
            bands  = total_score_bands(st.max_score)
            colors = theme.total_pie_colors
        else:
            bands  = section_score_bands(st.max_score)
            colors = st.pie_colors

        labels = [b[0] for b in bands]

        plot_histogram(ax_hist, raw, st, theme, bands, colors)
        ax_hist.set_title("Distribution", fontsize=9, fontweight="bold",
                          color=theme.text, pad=8)

        plot_range_pie(ax_pie, raw, bands, labels, colors, theme)
        ax_pie.set_title("Score ranges", fontsize=9, fontweight="bold",
                         color=theme.text, pad=8)


# ---------------------------------------------------------------------------
# Main entry point  (no ExamConfig, no argparse)
# ---------------------------------------------------------------------------

def build_report_image(
    df: pd.DataFrame,
    output_path: Path,
    title: str = REPORT_TITLE,
    theme: DarkTheme | None = None,
) -> Path:
    """
    Generate the academic report PNG.

    Max scores are derived automatically from the data:
      - If df contains  max_mcq / max_complete / max_written / max_total  columns
        (or  choice_total_score / complete_total_score / written_total_score)
        those values are used as-is.
      - Otherwise the observed score ceiling is rounded up to the nearest
        standard value (10, 20, 25, 30 …).

    Required score columns: mcq, complete, written
    Optional columns that improve max inference: max_mcq, max_complete,
        max_written, max_total  (or their *_total_score aliases).
    """
    theme = theme or DarkTheme()
    _apply_rc(theme)

    maxes  = derive_max_scores(df)
    df     = enrich_dataframe(df, maxes)
    types  = build_score_types(maxes)

    fig = plt.figure(figsize=(16, 13), facecolor=theme.page_bg)
    fig.text(0.5, 0.975, title, ha="center", va="top",
             fontsize=22, fontweight="bold", color=theme.text)

    root = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.22, 1],
        left=0.05, right=0.97, top=0.93, bottom=0.04, hspace=0.30,
    )

    ax_perf = fig.add_subplot(root[0].subgridspec(1, 1)[0])
    _style_card_ax(ax_perf, theme, theme.text_muted)
    plot_section_performance(ax_perf, df, types, theme)
    ax_perf.set_title("Section performance", fontsize=10, fontweight="bold",
                      color=theme.text, pad=10)

    panels_gs = root[1].subgridspec(2, 2, hspace=0.38, wspace=0.26)
    for panel_gs, st in zip(panels_gs, types):
        TypePanel(st).render(fig, panel_gs, df, theme)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor=theme.page_bg)
    plt.close(fig)
    return output_path