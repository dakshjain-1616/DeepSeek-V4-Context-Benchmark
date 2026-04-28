"""Render the SVG charts shipped with the context-bench README.

Pure stdlib -- no matplotlib. Outputs five hand-built bar charts under
``assets/charts/`` summarizing the live numbers from ``results/live_*.json``
and the live-OpenRouter pricing snapshot.
"""

from __future__ import annotations

from pathlib import Path

W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 90, 30, 60, 70
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B

NEUTRAL = "#4a90e2"
HIGHLIGHT = "#f5a623"
WARN = "#d9534f"
GOOD = "#3aa55a"
TEXT = "#1f2933"
GRID = "#e4e7eb"
AXIS = "#9aa5b1"


def _render(
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    y_min: float,
    y_max: float,
    y_label: str,
    value_fmt: str = "{:.1f}",
    colors: list[str] | None = None,
    legend: str | None = None,
) -> str:
    n = len(values)
    bar_w = PLOT_W / n * 0.55
    gap = (PLOT_W / n) - bar_w
    y_ticks = [y_min + (y_max - y_min) * i / 4 for i in range(5)]

    def y_pix(v: float) -> float:
        return PAD_T + PLOT_H * (1.0 - (v - y_min) / (y_max - y_min))

    base_y = y_pix(y_min)
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    parts.append(
        f'<text x="{W/2}" y="28" text-anchor="middle" font-size="18" font-weight="600" fill="{TEXT}">{title}</text>'
    )
    parts.append(
        f'<text x="{W/2}" y="48" text-anchor="middle" font-size="12" fill="{AXIS}">{subtitle}</text>'
    )
    parts.append(
        f'<text x="22" y="{PAD_T + PLOT_H/2}" text-anchor="middle" font-size="12" fill="{AXIS}" '
        f'transform="rotate(-90 22 {PAD_T + PLOT_H/2})">{y_label}</text>'
    )
    for t in y_ticks:
        y = y_pix(t)
        parts.append(f'<line x1="{PAD_L}" y1="{y}" x2="{W-PAD_R}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
        tick_label = f"{int(t)}" if t == int(t) and y_max >= 100 else f"{t:.1f}"
        parts.append(
            f'<text x="{PAD_L-8}" y="{y+4}" text-anchor="end" font-size="11" fill="{AXIS}">{tick_label}</text>'
        )
    parts.append(f'<line x1="{PAD_L}" y1="{base_y}" x2="{W-PAD_R}" y2="{base_y}" stroke="{AXIS}" stroke-width="1.5"/>')

    for i, (label, v) in enumerate(zip(labels, values, strict=True)):
        x = PAD_L + i * (bar_w + gap) + gap / 2
        top_y = y_pix(v)
        color = (colors or [NEUTRAL] * n)[i]
        parts.append(
            f'<rect x="{x}" y="{top_y}" width="{bar_w}" height="{base_y - top_y}" fill="{color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{x + bar_w/2}" y="{top_y - 6}" text-anchor="middle" font-size="12" font-weight="600" fill="{TEXT}">{value_fmt.format(v)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w/2}" y="{base_y + 22}" text-anchor="middle" font-size="13" fill="{TEXT}">{label}</text>'
        )

    if legend:
        parts.append(
            f'<text x="{PAD_L}" y="{H-18}" font-size="12" fill="{AXIS}">{legend}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    out = Path("assets/charts")
    out.mkdir(parents=True, exist_ok=True)

    (out / "niah_depth.svg").write_text(
        _render(
            title="NIAH accuracy vs context length",
            subtitle="deepseek/deepseek-v4-flash - n=3 (~93K, ~187K) - n=2 (~968K)",
            labels=["~93K ctx", "~187K ctx", "~968K ctx"],
            values=[100.0, 100.0, 50.0],
            y_min=0,
            y_max=100,
            y_label="accuracy %",
            value_fmt="{:.0f}%",
            colors=[GOOD, GOOD, HIGHLIGHT],
            legend="v4-pro and llama-4-scout both fall to 0 % at ~968K -- see Findings.",
        )
    )

    (out / "cross_corpus.svg").write_text(
        _render(
            title="Accuracy across corpora - small-context runs",
            subtitle="3-model average - NIAH-93K, MultiHop, Codebase, Synthesis",
            labels=["NIAH-93K", "MultiHop", "Codebase", "Synthesis"],
            values=[89.0, 100.0, 78.0, 100.0],
            y_min=0,
            y_max=100,
            y_label="accuracy %",
            value_fmt="{:.0f}%",
            colors=[NEUTRAL, GOOD, NEUTRAL, GOOD],
        )
    )

    (out / "latency.svg").write_text(
        _render(
            title="Average latency at ~93K context",
            subtitle="NIAH, n=3 per model, OpenRouter",
            labels=["v4-flash", "v4-pro", "llama-4-scout"],
            values=[6.9, 11.6, 7.4],
            y_min=0,
            y_max=14,
            y_label="latency (s)",
            value_fmt="{:.1f}s",
            colors=[GOOD, HIGHLIGHT, NEUTRAL],
        )
    )

    (out / "token_budget.svg").write_text(
        _render(
            title="Per-task input tokens by corpus (typical)",
            subtitle="What you'll send to the API per task at default settings",
            labels=["NIAH-100K", "NIAH-500K", "NIAH-1M", "MultiHop-500K", "Codebase-500K", "Synthesis-10doc"],
            values=[100, 500, 1000, 500, 500, 200],
            y_min=0,
            y_max=1100,
            y_label="K tokens",
            value_fmt="{:.0f}K",
            colors=[NEUTRAL, NEUTRAL, WARN, NEUTRAL, NEUTRAL, NEUTRAL],
        )
    )

    # Cost per 100 tasks at 100K input tokens, recomputed with live OpenRouter
    # prices (2026-04-28) and a 99/1 input/output split.
    # v4-flash: 10M × ($0.14×.99 + $0.28×.01) = $1.414
    # v4-pro:   10M × ($0.435×.99 + $0.87×.01) = $4.396
    # scout:    10M × ($0.08×.99 + $0.30×.01) = $0.822
    (out / "cost_per_100.svg").write_text(
        _render(
            title="USD per 100 tasks at 100K input tokens / task",
            subtitle="Live OpenRouter pricing snapshot - 2026-04-28",
            labels=["v4-flash", "v4-pro", "llama-4-scout"],
            values=[1.41, 4.40, 0.82],
            y_min=0,
            y_max=5,
            y_label="USD",
            value_fmt="${:.2f}",
            colors=[NEUTRAL, HIGHLIGHT, GOOD],
        )
    )

    # Also render PNG copies — HuggingFace's markdown sanitizer drops <img>
    # refs to *.svg, and a few corp GitHub deployments do too. We keep both
    # formats so the README renders crisp on every surface.
    try:
        import cairosvg  # type: ignore[import-not-found]
    except ImportError:
        print(f"wrote {len(list(out.glob('*.svg')))} SVG files to {out}/  (skip PNG: install cairosvg to enable)")
        return
    for svg in out.glob("*.svg"):
        cairosvg.svg2png(url=str(svg), write_to=str(svg.with_suffix(".png")), output_width=1440)
    print(f"wrote {len(list(out.glob('*.svg')))} SVG + PNG pairs to {out}/")


if __name__ == "__main__":
    main()
