"""Generate colorful infographics for the DeepSeek V4 Context Benchmark."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "assets" / "infographics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0D1117"
BG2     = "#161B22"
BG3     = "#21262D"
BORDER  = "#30363D"
TEXT    = "#E6EDF3"
MUTED   = "#8B949E"
FLASH   = "#58A6FF"   # DeepSeek Flash  – vivid blue
PRO     = "#BC8CFF"   # DeepSeek Pro    – vivid purple
LLAMA   = "#3FB950"   # Llama 4 Scout   – vivid green
NIAH_C  = "#FF7B72"   # NIAH            – coral
MULTI_C = "#FFA657"   # MultiHop        – orange
CODE_C  = "#79C0FF"   # Codebase        – sky
SYNTH_C = "#D2A8FF"   # Synthesis       – lavender
ACCENT  = "#F78166"   # accent / warning


def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG2,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "grid.color": BORDER,
        "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans",
        "legend.facecolor": BG3,
        "legend.edgecolor": BORDER,
        "legend.labelcolor": TEXT,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 1.  BENCHMARK RESULTS DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

def make_benchmark_dashboard():
    set_dark_style()
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs = GridSpec(3, 3, figure=fig,
                  left=0.06, right=0.97,
                  top=0.88, bottom=0.07,
                  hspace=0.55, wspace=0.38)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.945, "DeepSeek V4 Context Benchmark — Live Results",
             ha="center", va="center", fontsize=19, fontweight="bold",
             color=TEXT)
    fig.text(0.5, 0.915, "OpenRouter · April 2026  |  contains-match scorer  |  $0.90 total spend across 51 live API calls",
             ha="center", va="center", fontsize=10, color=MUTED)

    # ── Stat cards (top row) ──────────────────────────────────────────────────
    cards = [
        ("$0.90",    "Total API Spend",        FLASH),
        ("51",       "Live API Calls",          PRO),
        ("3 Models", "Compared",               LLAMA),
        ("4 Corpora","Evaluated",              NIAH_C),
        ("1M Tokens","Max Context Window",     MULTI_C),
    ]
    ax_cards = fig.add_axes([0.0, 0.87, 1.0, 0.0])
    ax_cards.set_axis_off()

    card_w, card_h = 0.175, 0.072
    xs = np.linspace(0.065, 0.835, 5)
    for (val, lbl, col), x in zip(cards, xs):
        fancy = FancyBboxPatch((x, -0.01), card_w, card_h,
                               boxstyle="round,pad=0.01",
                               linewidth=1.2, edgecolor=col,
                               facecolor=BG3, transform=ax_cards.transAxes,
                               clip_on=False)
        ax_cards.add_patch(fancy)
        ax_cards.text(x + card_w / 2, card_h * 0.68, val,
                      ha="center", va="center", fontsize=15,
                      fontweight="bold", color=col,
                      transform=ax_cards.transAxes)
        ax_cards.text(x + card_w / 2, card_h * 0.20, lbl,
                      ha="center", va="center", fontsize=8,
                      color=MUTED, transform=ax_cards.transAxes)

    # ── Grouped bar: accuracy by corpus & model (main chart) ─────────────────
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    corpora = ["NIAH\n93K", "NIAH\n187K", "NIAH\n968K",
               "MultiHop", "Codebase", "Synthesis"]
    flash_acc  = [100, 100,  50, 100,  67, 100]
    pro_acc    = [100, 100,   0, 100,  67, 100]
    llama_acc  = [ 67,  67,   0, 100, 100, 100]

    x = np.arange(len(corpora))
    w = 0.26
    b1 = ax1.bar(x - w, flash_acc, w, label="v4-flash",     color=FLASH,  alpha=0.92, zorder=3)
    b2 = ax1.bar(x,     pro_acc,   w, label="v4-pro",       color=PRO,    alpha=0.92, zorder=3)
    b3 = ax1.bar(x + w, llama_acc, w, label="llama-4-scout",color=LLAMA,  alpha=0.92, zorder=3)

    # Value labels
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1,
                     f"{int(h)}%", ha="center", va="bottom",
                     fontsize=7.5, color=TEXT, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(corpora, fontsize=9)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax1.set_ylim(0, 115)
    ax1.set_title("Accuracy by Corpus & Model", fontsize=11, fontweight="bold",
                  color=TEXT, pad=8)
    ax1.axhline(100, color=BORDER, linewidth=0.8, linestyle="--", zorder=2)
    ax1.grid(axis="y", zorder=1)
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax1.tick_params(length=0)

    # ── NIAH depth line chart ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ctx_sizes = [93, 187, 968]
    ax2.plot(ctx_sizes, [100, 100, 50],  marker="o", color=FLASH, lw=2.2, label="v4-flash")
    ax2.plot(ctx_sizes, [100, 100,  0],  marker="s", color=PRO,   lw=2.2, label="v4-pro")
    ax2.plot(ctx_sizes, [ 67,  67,  0],  marker="^", color=LLAMA, lw=2.2, label="llama")
    ax2.fill_between(ctx_sizes, [100, 100, 50], 0, alpha=0.12, color=FLASH)
    ax2.set_title("NIAH: Accuracy vs Context", fontsize=10,
                  fontweight="bold", color=TEXT, pad=6)
    ax2.set_xlabel("Input Tokens (K)", fontsize=8, color=MUTED)
    ax2.set_yticks([0, 50, 100])
    ax2.set_yticklabels(["0%", "50%", "100%"], fontsize=8)
    ax2.set_ylim(-5, 115)
    ax2.set_xticks(ctx_sizes)
    ax2.set_xticklabels(["93K", "187K", "968K"], fontsize=8)
    ax2.legend(fontsize=7.5, loc="lower left")
    ax2.grid(zorder=1)
    ax2.tick_params(length=0)

    # ── Latency bars (93K NIAH) ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    models_short = ["v4-flash", "v4-pro", "llama"]
    latencies = [6.9, 11.6, 7.4]
    bars = ax3.barh(models_short, latencies,
                    color=[FLASH, PRO, LLAMA], alpha=0.9, height=0.5, zorder=3)
    for bar, val in zip(bars, latencies):
        ax3.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
                 f"{val}s", va="center", fontsize=9,
                 fontweight="bold", color=TEXT)
    ax3.set_title("Latency at NIAH-93K (s)", fontsize=10,
                  fontweight="bold", color=TEXT, pad=6)
    ax3.set_xlim(0, 17)
    ax3.grid(axis="x", zorder=1)
    ax3.tick_params(length=0)

    # ── Cost comparison (per 100 tasks, 100K tokens) ──────────────────────────
    ax4 = fig.add_subplot(gs[2, 0:2])
    categories = ["NIAH-93K", "NIAH-187K", "MultiHop", "Codebase", "Synthesis"]
    # costs per task × 100  (from README: $0.040, $0.079, $0.0001, $0.0065, $0.0019)
    cost_flash = [4.0,  7.9,  0.01, 0.65, 0.19]
    cost_pro   = [12.3, 24.6, 0.05, 2.1,  0.60]
    cost_llama = [2.3,  4.6,  0.01, 0.38, 0.11]

    x4 = np.arange(len(categories))
    ax4.bar(x4 - w, cost_flash, w, label="v4-flash",      color=FLASH, alpha=0.92, zorder=3)
    ax4.bar(x4,     cost_pro,   w, label="v4-pro",        color=PRO,   alpha=0.92, zorder=3)
    ax4.bar(x4 + w, cost_llama, w, label="llama-4-scout", color=LLAMA, alpha=0.92, zorder=3)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.set_ylabel("USD per 100 Tasks", fontsize=9, color=MUTED)
    ax4.set_title("Cost per 100 Tasks by Model (live OpenRouter pricing)", fontsize=10,
                  fontweight="bold", color=TEXT, pad=6)
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", zorder=1)
    ax4.tick_params(length=0)

    # ── Pricing table ─────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_axis_off()
    ax5.set_title("Live Pricing ($/1M tokens)", fontsize=10,
                  fontweight="bold", color=TEXT, pad=6)
    rows = [
        ("v4-flash", "$0.14", "$0.28"),
        ("v4-pro",   "$0.435", "$0.87"),
        ("llama",    "$0.08", "$0.30"),
    ]
    colors_row = [FLASH, PRO, LLAMA]
    col_labels = ["Model", "Input", "Output"]
    for ci, lbl in enumerate(col_labels):
        ax5.text(0.1 + ci * 0.3, 0.92, lbl, ha="left", va="top",
                 fontsize=9, fontweight="bold", color=MUTED,
                 transform=ax5.transAxes)
    for ri, (model, inp, out) in enumerate(rows):
        y = 0.72 - ri * 0.24
        col = colors_row[ri]
        ax5.add_patch(FancyBboxPatch((0.02, y - 0.06), 0.96, 0.20,
                                      boxstyle="round,pad=0.01",
                                      linewidth=1, edgecolor=col,
                                      facecolor=BG3,
                                      transform=ax5.transAxes, clip_on=False))
        ax5.text(0.10, y + 0.05, model, ha="left", va="center",
                 fontsize=9, fontweight="bold", color=col,
                 transform=ax5.transAxes)
        ax5.text(0.40, y + 0.05, inp, ha="left", va="center",
                 fontsize=9, color=TEXT, transform=ax5.transAxes)
        ax5.text(0.70, y + 0.05, out, ha="left", va="center",
                 fontsize=9, color=TEXT, transform=ax5.transAxes)

    path = OUTPUT_DIR / "benchmark_results.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  TEST COVERAGE INFOGRAPHIC
# ──────────────────────────────────────────────────────────────────────────────

def make_testing_infographic():
    set_dark_style()
    fig = plt.figure(figsize=(16, 8), facecolor=BG)
    gs = GridSpec(2, 3, figure=fig,
                  left=0.05, right=0.97,
                  top=0.88, bottom=0.07,
                  hspace=0.5, wspace=0.38)

    # Title
    fig.text(0.5, 0.945, "DeepSeek V4 Context Benchmark — Test Quality",
             ha="center", fontsize=18, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.915, "221 tests · 92% coverage · 4 × 20 benchmark corpus suites · zero tautological assertions",
             ha="center", fontsize=10, color=MUTED)

    # ── Coverage donut ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    covered, uncov = 92, 8
    wedges, _ = ax1.pie([covered, uncov],
                        colors=[FLASH, BG3],
                        startangle=90, counterclock=False,
                        wedgeprops=dict(width=0.45, edgecolor=BG, linewidth=2))
    ax1.text(0, 0.05, "92%", ha="center", va="center",
             fontsize=26, fontweight="bold", color=FLASH)
    ax1.text(0, -0.22, "Coverage", ha="center", va="center",
             fontsize=11, color=MUTED)
    ax1.set_title("Overall Test Coverage", fontsize=11,
                  fontweight="bold", color=TEXT, pad=10)

    # ── Test count bar per module ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0:2, 1])
    modules = [
        "test_scorer",
        "test_client",
        "test_corpora_niah",
        "test_corpora_multihop",
        "test_corpora_codebase",
        "test_corpora_synthesis",
        "test_tokenizer",
        "test_cli",
        "test_config",
        "test_card",
        "test_report",
        "test_runner",
        "test_integration",
    ]
    counts = [33, 23, 20, 20, 20, 20, 19, 15, 13, 13, 12, 12, 1]
    colors = ([ACCENT] +           # scorer
              [MUTED] +            # client
              [NIAH_C, MULTI_C, CODE_C, SYNTH_C] +   # corpus 4×20
              [MUTED] * 7)         # rest

    y_pos = np.arange(len(modules))
    bars = ax2.barh(y_pos, counts, color=colors, alpha=0.9,
                    height=0.65, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(modules, fontsize=8.5)
    ax2.invert_yaxis()
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(cnt), va="center", fontsize=9,
                 fontweight="bold", color=TEXT)
    ax2.set_xlim(0, 42)
    ax2.set_xlabel("Number of Tests", fontsize=9, color=MUTED)
    ax2.set_title("Tests per Module (Total: 221)", fontsize=11,
                  fontweight="bold", color=TEXT, pad=8)
    ax2.grid(axis="x", zorder=1)
    ax2.tick_params(length=0)

    # ── Per-module coverage breakdown ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cov_modules = ["config", "card", "niah", "codebase",
                   "report", "tokenizer", "scorer",
                   "client", "multihop", "synthesis",
                   "runner", "cli"]
    cov_vals = [100, 100, 100, 100, 98, 97, 94, 93, 95, 91, 82, 82]
    colors3 = [LLAMA if v >= 95 else FLASH if v >= 85 else ACCENT
               for v in cov_vals]

    x3 = np.arange(len(cov_modules))
    ax3.bar(x3, cov_vals, color=colors3, alpha=0.9, width=0.65, zorder=3)
    ax3.axhline(85, color=ACCENT, linewidth=1.4, linestyle="--",
                zorder=4, label="85% threshold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(cov_modules, rotation=40, ha="right", fontsize=8)
    ax3.set_yticks([70, 80, 85, 90, 95, 100])
    ax3.set_yticklabels(["70%", "80%", "85%", "90%", "95%", "100%"], fontsize=8)
    ax3.set_ylim(70, 106)
    ax3.set_title("Coverage by Source Module", fontsize=10,
                  fontweight="bold", color=TEXT, pad=8)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", zorder=1)
    ax3.tick_params(length=0)

    # ── 4 corpus cards ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_axis_off()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title("Corpus Suite Highlights", fontsize=10,
                  fontweight="bold", color=TEXT, pad=8)

    corpus_info = [
        (NIAH_C,  "NIAH",      "20 tests",
         "Code planted in haystack\nExact position tracking\nCode length variations"),
        (MULTI_C, "MultiHop",  "20 tests",
         "Chain linkage validated\nRequired keys enforced\nSeed determinism"),
        (CODE_C,  "Codebase",  "20 tests",
         "Named patterns checked\nFile markers verified\nAll 6 languages"),
        (SYNTH_C, "Synthesis", "20 tests",
         "Planted marker in content\nJSON structure asserted\nSeed variability"),
    ]
    for i, (col, name, cnt, desc) in enumerate(corpus_info):
        row, col_idx = divmod(i, 2)
        x0 = 0.02 + col_idx * 0.50
        y0 = 0.52 - row * 0.50
        ax4.add_patch(FancyBboxPatch((x0, y0), 0.46, 0.44,
                                      boxstyle="round,pad=0.015",
                                      linewidth=1.5, edgecolor=col,
                                      facecolor=BG3,
                                      transform=ax4.transAxes, clip_on=False))
        ax4.text(x0 + 0.23, y0 + 0.35, name,
                 ha="center", fontsize=10, fontweight="bold",
                 color=col, transform=ax4.transAxes)
        ax4.text(x0 + 0.23, y0 + 0.23, cnt,
                 ha="center", fontsize=9, color=TEXT,
                 transform=ax4.transAxes)
        ax4.text(x0 + 0.23, y0 + 0.05, desc,
                 ha="center", va="center", fontsize=7,
                 color=MUTED, transform=ax4.transAxes,
                 multialignment="center")

    # ── Assertion quality legend ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_axis_off()
    ax5.set_title("Assertion Quality Fixes Applied", fontsize=10,
                  fontweight="bold", color=TEXT, pad=8)
    fixes = [
        (LLAMA,  "Removed tautological `or len(x) > 0`"),
        (LLAMA,  "Exact hop count: `== hops` for 1- and 2-hop"),
        (LLAMA,  "Named patterns: `calculate_sum` / `DataProcessor`"),
        (LLAMA,  "Narrative: `Once upon a time` (single check)"),
        (LLAMA,  "Dialogue: `'\"' and 'said'` (both required)"),
        (LLAMA,  "Structured: `'[' and '{'` (both required)"),
        (LLAMA,  "Card: exact strings — `Multi-hop Reasoning` etc."),
        (LLAMA,  "Report: `80.00%` exact (not `or 0.8`)"),
        (LLAMA,  "Tokenizer: `len == 1` (not `>= 1`)"),
        (LLAMA,  "CLI: both `deepseek and llama` required"),
        (FLASH,  "3-hop documented limitation (structurally impossible)"),
    ]
    for i, (col, text) in enumerate(fixes):
        y = 0.93 - i * 0.086
        ax5.add_patch(Circle((0.035, y), 0.018,
                              color=col, transform=ax5.transAxes,
                              clip_on=False))
        ax5.text(0.07, y, text, va="center", fontsize=8,
                 color=TEXT, transform=ax5.transAxes)

    path = OUTPUT_DIR / "testing_overview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CORPUS TYPES INFOGRAPHIC
# ──────────────────────────────────────────────────────────────────────────────

def make_corpus_infographic():
    set_dark_style()
    fig = plt.figure(figsize=(16, 9), facecolor=BG)

    fig.text(0.5, 0.96, "DeepSeek V4 Context Benchmark — Four Corpus Types",
             ha="center", fontsize=18, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.935,
             "Each corpus type tests a distinct dimension of long-context LLM capability",
             ha="center", fontsize=10, color=MUTED)

    # ── Layout: 4 corpus panels ───────────────────────────────────────────────
    corpus_data = [
        {
            "title": "NIAH",
            "subtitle": "Needle In A Haystack",
            "color": NIAH_C,
            "icon_label": "🔍",
            "what": "Embeds a secret uppercase code inside\nthousands of sentences. Model must retrieve\nthe exact code from anywhere in the context.",
            "key_features": [
                "1–3 needles per sample",
                "Configurable code length (4–12 chars)",
                "Position: start / middle / end / random",
                "Haystack scales to 1M+ tokens",
            ],
            "acc": [100, 100, 50],
            "acc_labels": ["93K", "187K", "968K"],
            "stat_label": "100% at <200K ctx",
        },
        {
            "title": "MultiHop",
            "subtitle": "Multi-step Reasoning",
            "color": MULTI_C,
            "icon_label": "🔗",
            "what": "Generates entity graphs where answers\nrequire chaining 1–3 facts scattered\nacross the document.",
            "key_features": [
                "Validated chain linkage (A→B→C)",
                "Required keys: fact, from_entity,\n  to_entity, relationship",
                "Answer = last chain target entity",
                "Configurable hop depth",
            ],
            "acc": [100, 100, 100],
            "acc_labels": ["Flash", "Pro", "Llama"],
            "stat_label": "100% across all models\n(small ctx)",
        },
        {
            "title": "Codebase",
            "subtitle": "Code Pattern Analysis",
            "color": CODE_C,
            "icon_label": "💻",
            "what": "Synthesises a multi-file code repository\n(up to 20 files) and asks the model\nto locate specific named functions.",
            "key_features": [
                "6 languages: Python, JS, TS, Java, C++, Rust",
                "Named patterns: calculate_sum,\n  DataProcessor, find_max …",
                "Per-file section headers (// File:)",
                "Pattern location index tracked",
            ],
            "acc": [67, 67, 100],
            "acc_labels": ["Flash", "Pro", "Llama"],
            "stat_label": "Llama outperforms\nDeepSeek on code",
        },
        {
            "title": "Synthesis",
            "subtitle": "Synthetic Data Retrieval",
            "color": SYNTH_C,
            "icon_label": "🧬",
            "what": "Generates narrative / dialogue / structured\ncontent with a planted 8-char hex marker.\nModel must find the exact marker.",
            "key_features": [
                "Marker physically in content (verified)",
                "Narrative: templates with vocabulary",
                "Dialogue: multi-turn with speakers",
                "Structured: JSON arrays of objects",
            ],
            "acc": [100, 100, 100],
            "acc_labels": ["Flash", "Pro", "Llama"],
            "stat_label": "100% across all models",
        },
    ]

    panel_xs = [0.01, 0.26, 0.51, 0.76]
    panel_w = 0.225
    panel_y = 0.06
    panel_h = 0.84

    for data, px in zip(corpus_data, panel_xs):
        col = data["color"]
        ax = fig.add_axes([px, panel_y, panel_w, panel_h])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.patch.set_facecolor(BG2)

        # Panel border
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_edgecolor(col)
            ax.spines[spine].set_linewidth(1.8)

        # Header
        ax.add_patch(FancyBboxPatch((0, 0.88), 1, 0.12,
                                     boxstyle="square",
                                     facecolor=col, alpha=0.18,
                                     edgecolor="none",
                                     transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 0.935, data["title"],
                ha="center", va="center", fontsize=18,
                fontweight="bold", color=col,
                transform=ax.transAxes)
        ax.text(0.5, 0.895, data["subtitle"],
                ha="center", va="center", fontsize=9,
                color=TEXT, transform=ax.transAxes)

        # What it tests
        ax.text(0.5, 0.84, "What it tests",
                ha="center", fontsize=9, fontweight="bold",
                color=MUTED, transform=ax.transAxes)
        ax.text(0.5, 0.76, data["what"],
                ha="center", va="top", fontsize=8.2,
                color=TEXT, transform=ax.transAxes,
                multialignment="center")

        # Divider
        ax.axhline(0.64, color=BORDER, linewidth=0.8, xmin=0.05, xmax=0.95)

        # Key features
        ax.text(0.07, 0.62, "Key Features",
                fontsize=9, fontweight="bold",
                color=MUTED, transform=ax.transAxes)
        for i, feat in enumerate(data["key_features"]):
            ax.text(0.07, 0.57 - i * 0.09, f"▸  {feat}",
                    fontsize=8, color=TEXT,
                    transform=ax.transAxes, va="top")

        # Divider
        ax.axhline(0.20, color=BORDER, linewidth=0.8, xmin=0.05, xmax=0.95)

        # Mini accuracy chart
        xs = np.arange(len(data["acc"]))
        bar_h = [v / 100 * 0.17 for v in data["acc"]]
        bar_w = 0.20
        bar_bottom = 0.025
        for xi, (bh, label) in enumerate(zip(bar_h, data["acc_labels"])):
            bx = 0.10 + xi * 0.30
            bar_col = LLAMA if data["acc"][xi] == 100 else (FLASH if data["acc"][xi] >= 50 else ACCENT)
            rect = plt.Rectangle((bx, bar_bottom), bar_w, bh,
                                  facecolor=bar_col, alpha=0.9,
                                  transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            ax.text(bx + bar_w / 2, bar_bottom + bh + 0.008,
                    f"{data['acc'][xi]}%",
                    ha="center", fontsize=7.5, fontweight="bold",
                    color=TEXT, transform=ax.transAxes)
            ax.text(bx + bar_w / 2, bar_bottom - 0.015,
                    label, ha="center", fontsize=7,
                    color=MUTED, transform=ax.transAxes)

        ax.text(0.5, 0.225, data["stat_label"],
                ha="center", fontsize=8, color=col,
                fontweight="bold", transform=ax.transAxes,
                multialignment="center")

    path = OUTPUT_DIR / "corpus_types.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MODEL COMPARISON SPIDER / RADAR CHART
# ──────────────────────────────────────────────────────────────────────────────

def make_radar_chart():
    set_dark_style()
    fig = plt.figure(figsize=(14, 7), facecolor=BG)

    fig.text(0.5, 0.97, "Model Comparison — DeepSeek V4 vs Llama 4 Scout",
             ha="center", fontsize=17, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.945,
             "Radar across 6 dimensions  ·  Latency and cost axes are inverted (lower = better)",
             ha="center", fontsize=9.5, color=MUTED)

    categories = ["NIAH\nAccuracy", "MultiHop\nAccuracy", "Codebase\nAccuracy",
                  "Synthesis\nAccuracy", "Speed\n(inv. latency)", "Cost\n(inv. price)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Normalised values (0–1, higher = better)
    # Accuracy: average across context sizes
    # Speed: 1 - (latency / max_latency)  max=14.2s
    # Cost: 1 - (input_price / max_price)  max=$0.435
    flash_vals = [
        np.mean([100, 100, 50]) / 100,   # NIAH avg
        1.00,                             # MultiHop
        0.67,                             # Codebase
        1.00,                             # Synthesis
        1 - 3.6 / 14.2,                  # Speed (codebase latency 3.6s)
        1 - 0.14 / 0.435,                # Cost (input)
    ]
    pro_vals = [
        np.mean([100, 100, 0]) / 100,
        1.00,
        0.67,
        1.00,
        1 - 14.2 / 14.2,
        1 - 0.435 / 0.435,
    ]
    llama_vals = [
        np.mean([67, 67, 0]) / 100,
        1.00,
        1.00,
        1.00,
        1 - 2.4 / 14.2,
        1 - 0.08 / 0.435,
    ]

    def add_radar(ax, values, color, label):
        v = values + values[:1]
        ax.plot(angles, v, color=color, lw=2.2, label=label)
        ax.fill(angles, v, color=color, alpha=0.15)

    ax = fig.add_axes([0.05, 0.08, 0.52, 0.82], polar=True)
    ax.set_facecolor(BG2)

    # Grid rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [r] * (N + 1), color=BORDER, lw=0.8, zorder=1)
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1], color=BORDER, lw=0.8)

    add_radar(ax, flash_vals, FLASH, "v4-flash")
    add_radar(ax, pro_vals,   PRO,   "v4-pro")
    add_radar(ax, llama_vals, LLAMA, "llama-4-scout")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, color=TEXT)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color=MUTED)
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_edgecolor(BORDER)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # ── Side: head-to-head stat table ─────────────────────────────────────────
    ax2 = fig.add_axes([0.60, 0.08, 0.38, 0.82])
    ax2.set_axis_off()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    headers = ["Metric", "v4-flash", "v4-pro", "llama"]
    hcols   = [TEXT,     FLASH,      PRO,      LLAMA]
    rows = [
        ["Overall Accuracy",  "83%",   "72%",   "78%"],
        ["NIAH (small ctx)",  "100%",  "100%",  "67%"],
        ["NIAH (968K)",       "50%",   "0%",    "0%"],
        ["Codebase",          "67%",   "67%",   "100%"],
        ["Avg Latency (s)",   "6.9",   "11.6",  "7.4"],
        ["Input $/1M",        "$0.14", "$0.435","$0.08"],
        ["Output $/1M",       "$0.28", "$0.87", "$0.30"],
        ["Context Window",    "1M",    "1M",    "1M"],
        ["Best At",           "Speed\n& cost",
                              "Large\nmodels",
                              "Code\nanalysis"],
    ]
    row_h = 0.086
    col_xs = [0.01, 0.34, 0.57, 0.78]

    ax2.text(0.5, 0.975, "Head-to-Head Summary",
             ha="center", fontsize=12, fontweight="bold",
             color=TEXT, transform=ax2.transAxes)

    for ci, (hdr, hcol) in enumerate(zip(headers, hcols)):
        ax2.text(col_xs[ci] + 0.01, 0.925, hdr, fontsize=9.5,
                 fontweight="bold", color=hcol,
                 transform=ax2.transAxes)
    ax2.axhline(0.90, color=BORDER, linewidth=0.8)

    for ri, row in enumerate(rows):
        y = 0.875 - ri * row_h
        bg_col = BG3 if ri % 2 == 0 else BG2
        ax2.add_patch(FancyBboxPatch((0, y - row_h * 0.45), 1.0, row_h * 0.88,
                                      boxstyle="square",
                                      facecolor=bg_col, edgecolor="none",
                                      transform=ax2.transAxes, clip_on=False))
        for ci, val in enumerate(row):
            c = TEXT if ci == 0 else [FLASH, PRO, LLAMA][ci - 1]
            ax2.text(col_xs[ci] + 0.01, y - row_h * 0.02, val,
                     fontsize=8.5, color=c if ci > 0 else MUTED,
                     va="center",
                     transform=ax2.transAxes,
                     multialignment="center")

    path = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    make_benchmark_dashboard()
    make_testing_infographic()
    make_corpus_infographic()
    make_radar_chart()
    print("\nAll infographics generated successfully.")
