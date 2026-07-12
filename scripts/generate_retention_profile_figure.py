"""Generate the singleton-retention profile figure used by the manuscript.

The plot is deterministic and uses only analytic formulas from the paper.
"""

from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def block_divergence(q: float, p: float) -> float:
    """KL(Bernoulli(q) || Bernoulli(p)), with q,p in (0,1)."""
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def multiplicity_profile(
    epsilon: float = 0.08,
    k: int = 256,
    n_spectators: int = 300,
    delta: float = 0.005,
) -> tuple[float, float, float, float]:
    """Return normalized A/B row heights, L1, and R(A)."""
    d_a = block_divergence(1.0 - delta, epsilon)
    alpha = epsilon * delta / (1.0 - epsilon)
    d_b = block_divergence(alpha, epsilon)
    r_a = d_a / math.log(k / epsilon)
    r_b = d_b / math.log(n_spectators / (1.0 - epsilon))
    m_1 = max(r_a, r_b)
    l_1 = (epsilon * r_a + (1.0 - epsilon) * r_b) / m_1
    r_set = d_a / math.log(1.0 / epsilon)
    return r_a / m_1, r_b / m_1, l_1, r_set


def draw_profile(
    pdf: canvas.Canvas,
    x0: float,
    y0: float,
    width: float,
    height: float,
    widths: list[float],
    heights: list[float],
    title: str,
) -> None:
    plot_y = y0 + 38
    plot_h = height - 70
    plot_x = x0 + 30
    plot_w = width - 38

    pdf.setStrokeColor(HexColor("#d0d0d0"))
    pdf.setLineWidth(0.45)
    for value in (0.0, 0.5, 1.0):
        yy = plot_y + value * plot_h
        pdf.line(plot_x, yy, plot_x + plot_w, yy)

    left = 0.0
    colors = [HexColor("#1f5a7a"), HexColor("#9ecae1"), HexColor("#d9e8ef")]
    for idx, (mass, retention) in enumerate(zip(widths, heights)):
        pdf.setFillColor(colors[min(idx, len(colors) - 1)])
        pdf.setStrokeColor(white)
        pdf.setLineWidth(0.8)
        pdf.rect(
            plot_x + left * plot_w,
            plot_y,
            mass * plot_w,
            retention * plot_h,
            fill=1,
            stroke=1,
        )
        left += mass

    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setFillColor(HexColor("#222222"))
    pdf.setLineWidth(0.6)
    pdf.line(plot_x, plot_y, plot_x + plot_w, plot_y)
    pdf.line(plot_x, plot_y, plot_x, plot_y + plot_h)

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawCentredString(x0 + width / 2, y0 + height - 17, title)
    pdf.setFont("Helvetica", 10.5)
    for value, label in ((0.0, "0"), (0.5, ".5"), (1.0, "1")):
        xx = plot_x + value * plot_w
        pdf.drawCentredString(xx, plot_y - 14, label)
        yy = plot_y + value * plot_h
        pdf.drawRightString(plot_x - 4, yy - 2.5, label)
    pdf.drawCentredString(plot_x + plot_w / 2, y0 + 3, "cumulative stationary mass")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    mult_a, mult_b, mult_l, mult_r_set = multiplicity_profile()
    page_w, page_h = 666.0, 235.0
    pdf = canvas.Canvas(
        str(OUT / "singleton-retention-profiles.pdf"),
        pagesize=(page_w, page_h),
        invariant=1,
    )
    pdf.setTitle("Singleton retention profiles")
    pdf.setFillColor(HexColor("#222222"))
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawCentredString(
        page_w / 2,
        page_h - 17,
        "Area under the sorted profile = singleton localization index L1",
    )

    margin, gap = 26.0, 12.0
    panel_w = (page_w - 2 * margin - 2 * gap) / 3
    panel_h = page_h - 30
    pdf.saveState()
    pdf.setFont("Helvetica", 11)
    pdf.translate(11, page_h / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "normalized singleton retention  r(x) / M1")
    pdf.restoreState()
    draw_profile(pdf, margin, 7, panel_w, panel_h, [1.0], [1.0], "Lazy cycle")
    draw_profile(
        pdf,
        margin + panel_w + gap,
        7,
        panel_w,
        panel_h,
        [0.5, 0.5],
        [1.0, 0.0],
        "Lazy star",
    )
    draw_profile(
        pdf,
        margin + 2 * (panel_w + gap),
        7,
        panel_w,
        panel_h,
        [0.08, 0.92],
        [mult_a, mult_b],
        "Multiplicity construction",
    )

    pdf.save()

    print(
        f"multiplicity: normalized A={mult_a:.6f}, normalized B={mult_b:.6f}, "
        f"L1={mult_l:.6f}, R(A)={mult_r_set:.6f}"
    )


if __name__ == "__main__":
    main()
