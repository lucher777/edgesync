"""Draw the figures of the edgesync letter from ``paper/results/results_summary.json``.

Usage: ``python make_figures.py`` (writes PDF and PNG files to ``paper/figures``).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "results_summary.json"
FIG = HERE.parent / "figures"

# Okabe-Ito palette
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#666666")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 3.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def panel(ax, letter):
    ax.text(-0.20, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig1_architecture(r):
    fig, ax = plt.subplots(figsize=(3.45, 1.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, title, body, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    ec=ec, fc="white", lw=0.8))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=5.6,
                fontweight="bold")
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=4.9,
                linespacing=1.3)

    def arrow(x0, y0, x1, y1, c="black"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                     lw=0.8, color=c, shrinkA=0, shrinkB=0))

    k64 = r["sections_O3"]["sizeof_kernel_fp64_bytes"]
    ax.add_patch(FancyBboxPatch((19, 1), 56, 34.5, boxstyle="round,pad=0,rounding_size=1.5",
                                ec=BLUE, fc="#F3F8FC", lw=0.8, ls=(0, (3, 2))))
    ax.text(47, 39.5, f"one static struct, no heap ({k64:,} B in FP64)", ha="center",
            va="center", fontsize=5.4, color=BLUE)

    box(0.5, 20, 15.5, 11, "Samples", r"$(t_{\mathrm{loc}}, x)$", "black")
    box(0.5, 4, 15.5, 11, "Beacons", r"$(t_{\mathrm{loc}}, t_{\mathrm{ref}})$", "black")
    box(22, 19.5, 22, 13, "Ring buffer", "512 slots\nO(1) push", BLUE)
    box(22, 3.5, 22, 13, "Huber-RLS", "O(1) 2x2 update\nrejects bursts", ORANGE)
    box(49, 6.5, 23, 22, "Resampler", "cursor + binary\nsearch; cubic\nFarrow; linear\nacross gaps", GREEN)
    box(80, 12, 18, 11, "Output", r"$\hat{x}(t_{\mathrm{ref}})$", "black")

    arrow(16, 25.5, 22, 25.5)
    arrow(16, 9.5, 22, 9.5)
    arrow(44, 25.5, 49, 22.5, BLUE)
    arrow(44, 10.0, 49, 12.5, ORANGE)
    arrow(72, 17.5, 80, 17.5, GREEN)
    save(fig, "Fig1_architecture")


def fig2_results(r):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.95))
    fig.subplots_adjust(wspace=0.42)

    # a: Farrow error vs. bound
    ax = axes[0]
    rows = r["property2_oh4_scaling"]
    fs = [x["fs_hz"] for x in rows]
    ax.loglog(fs, [x["linear_bound"] for x in rows], color=ORANGE, ls="-", lw=2.4, alpha=0.35,
              label="Linear bound")
    ax.loglog(fs, [x["linear_max_err"] for x in rows], color=ORANGE, ls="none", marker="s",
              label="Linear, measured")
    ax.loglog(fs, [x["property2_bound"] for x in rows], color=BLUE, ls="-", lw=2.4, alpha=0.35,
              label="Cubic bound")
    ax.loglog(fs, [x["farrow_max_err"] for x in rows], color=BLUE, ls="none", marker="o",
              label="Cubic, measured")
    ax.set_xticks(fs)
    ax.set_xticklabels([f"{int(f)}" for f in fs])
    ax.minorticks_off()
    ax.set_xlabel("Sampling rate (Hz)")
    ax.set_ylabel("Max. interpolation error")
    ax.set_ylim(1e-8, 0.3)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.1)
    panel(ax, "a")

    # b: outlier sweep, PPG SNR
    ax = axes[1]
    sw = r["wesad_s10_ppg_outlier_sweep"]
    rate = [100 * x["outlier_rate"] for x in sw]
    for key, lab, c, m, ls in [("fp64", "edgesync", BLUE, "o", "-"),
                               ("hrls_linear", "Huber-RLS + linear", GREEN, "^", "--"),
                               ("std_rls_cubic", "Unweighted RLS", VERM, "s", "-"),
                               ("unsync", "Offset only", PURPLE, "D", ":")]:
        ax.plot(rate, [x[key]["snr_db"] for x in sw], color=c, marker=m, ls=ls, label=lab)
    ax.set_xticks(rate)
    ax.set_xlabel("Beacons with burst delay (%)")
    ax.set_ylabel("PPG reconstruction SNR (dB)")
    ax.set_ylim(0, 60)
    panel(ax, "b")

    # c: 5-subject beacon timing
    ax = axes[2]
    co = r["wesad_5subj_cohort"]
    ids = [x["subject_id"] for x in co]
    xs = range(len(ids))
    w = 0.26
    ax.bar([i - w for i in xs], [x["edgesync_med_ms"] for x in co], w, color=BLUE,
           label="edgesync", edgecolor="white", lw=0.4)
    ax.bar(list(xs), [x["std_rls_med_ms"] for x in co], w, color=VERM, label="Unweighted RLS",
           edgecolor="white", lw=0.4)
    ax.bar([i + w for i in xs], [x["unsync_med_ms"] for x in co], w, color=PURPLE,
           label="Offset only", edgecolor="white", lw=0.4)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 300)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(ids)
    ax.set_xlabel("WESAD subject (beacon schedule)")
    ax.set_ylabel("Median timing error (ms)")
    panel(ax, "c")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.66, 0.99), ncol=4,
               frameon=False, handlelength=2.0, columnspacing=1.2)
    save(fig, "Fig2_results")


def main():
    with RES.open(encoding="utf-8") as fh:
        r = json.load(fh)
    fig1_architecture(r)
    fig2_results(r)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
