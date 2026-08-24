"""Generates the compression/tractability right-triangle diagrams for
representations-entropy-tree.md. Hypotenuse lies on the y-axis (compression /
information); width = tractability. The two legs are annotated with arrows:
over-compression toward the top vertex (pure noise), under-compression toward
the root (too information dense).

The triangle is the *complete* envelope of the (compression, tractability)
plane: nothing exists outside it. The legs are maximum tractability by
construction, so a representation can only ever be a point inside.

Writes four files:
  ...tractability-triangle.png                - plain
  ...tractability-triangle-human.png          - "human understanding" wedge
  ...tractability-triangle-machine.png        - "machine understanding" wedge,
                                                x-axis = machine tractability
  ...tractability-triangle-representations.png - human vs machine *representations*
                                                on the human-tractability axes:
                                                human representations bulge to
                                                the apex, machine representations
                                                hug the low-tractability edge
Re-run to regenerate all four."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Wedge, Ellipse

H = 10.0                       # y-axis span (the hypotenuse)
W = H / 2.0                    # apex x: max-tractability point on the Thales circle
APEX = np.array([W, H / 2.0])
TOP = np.array([0.0, H])       # maximally compressed
ROOT = np.array([0.0, 0.0])    # Laplacian root

REGION_STYLES = {
    "yellow": dict(facecolor="#f3d96b", edgecolor="#caa12e", textcolor="#6b5410"),
    "blue": dict(facecolor="#a9c9e8", edgecolor="#3f7cb5", textcolor="#1f425f"),
}


def leg_arrow(ax, vertex, color, label):
    """Arrow running alongside a leg, from the apex toward `vertex`."""
    leg = vertex - APEX
    unit = leg / np.linalg.norm(leg)
    normal = np.array([unit[1], -unit[0]])            # outward normal
    if np.dot(normal, np.array([1.0, 0.0])) < 0:      # ensure it points right-ward
        normal = -normal
    tail = APEX + 0.12 * leg + 0.7 * normal
    head = APEX + 0.80 * leg + 0.7 * normal
    ax.add_patch(FancyArrowPatch(tail, head, arrowstyle="-|>", mutation_scale=20,
                                 color=color, lw=2.2, zorder=5))
    mid = APEX + 0.5 * leg + 1.55 * normal
    angle = np.degrees(np.arctan2(unit[1], unit[0]))
    if angle > 90 or angle < -90:
        angle += 180                                  # keep text upright
    ax.text(*mid, label, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=color, rotation=angle,
            rotation_mode="anchor", zorder=6)


def draw_wedge(ax, vertex, radius, theta1, theta2, label, color, label_pos,
               alpha=1.0):
    """A circular-sector region with a centred label (e.g. understanding zone)."""
    style = REGION_STYLES[color]
    ax.add_patch(Wedge(vertex, radius, theta1, theta2, alpha=alpha,
                       facecolor=style["facecolor"], edgecolor=style["edgecolor"],
                       lw=2.0, zorder=3))
    ax.text(*label_pos, label, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=style["textcolor"], zorder=6)


def draw_ellipse(ax, center, width, height, label, color, label_pos,
                 label_rotation=0, alpha=1.0):
    """An elliptical region with a (optionally rotated) centred label."""
    style = REGION_STYLES[color]
    ax.add_patch(Ellipse(center, width, height, alpha=alpha,
                         facecolor=style["facecolor"], edgecolor=style["edgecolor"],
                         lw=2.0, zorder=3))
    ax.text(*label_pos, label, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=style["textcolor"], rotation=label_rotation,
            zorder=6)


def build(out, wedge_label=None, wedge_color="yellow",
          wedge_label_pos=(APEX[0] - 1.45, APEX[1]),
          x_label="tractability", x_sublabel=None,
          ellipse_region=None,
          repr_label_pos=(W * 0.34, H * 0.22), repr_label_fontsize=13):
    fig, ax = plt.subplots(figsize=(8.0, 8.6))

    # --- the representation space (the complete envelope) -----------------
    usable = Polygon([ROOT, TOP, APEX], closed=True,
                     facecolor="#cde8d5", edgecolor="#2f7d4f", lw=2.0, zorder=2)
    ax.add_patch(usable)

    region_alpha = 0.8 if ellipse_region else 1.0

    # --- understanding / representation wedge around peak tractability ----
    if wedge_label:
        draw_wedge(ax, APEX, 2.6, 135, 225, wedge_label, wedge_color,
                   wedge_label_pos, alpha=region_alpha)

    # --- optional elliptical region (machine representations) -------------
    if ellipse_region:
        draw_ellipse(ax, alpha=region_alpha, **ellipse_region)

    # dashed line at the widest point (peak tractability)
    ax.plot([0, W], [H / 2, H / 2], ls="--", color="#2f7d4f", lw=1.2, zorder=4)

    ax.text(*repr_label_pos, "representation\nspace", ha="center", va="center",
            fontsize=repr_label_fontsize, fontweight="bold", color="#1f5b39",
            zorder=4)

    leg_arrow(ax, TOP, "#777777", "pure noise")
    leg_arrow(ax, ROOT, "#b5503c", "too information dense")

    # --- axes -------------------------------------------------------------
    ax.add_patch(FancyArrowPatch((0, 0), (0, H + 0.9), arrowstyle="-|>",
                 mutation_scale=18, color="black", lw=1.6, zorder=5))
    ax.add_patch(FancyArrowPatch((0, 0), (W + 1.0, 0), arrowstyle="-|>",
                 mutation_scale=18, color="black", lw=1.6, zorder=5))

    ax.text(W / 2 + 0.5, -0.7, x_label, ha="center", va="top",
            fontsize=13, fontweight="bold")
    if x_sublabel:
        ax.text(W / 2 + 0.5, -1.35, x_sublabel, ha="center", va="top",
                fontsize=10.5, style="italic", color="#444444")
    ax.text(-0.55, H / 2, "compression  /  information loss", ha="center",
            va="center", rotation=90, fontsize=13, fontweight="bold")

    # endpoints of the hypotenuse — drop the root label clear of the x-axis
    # sublabel when one is present
    root_xytext = (0.2, -2.15) if x_sublabel else (1.4, -1.7)
    ax.annotate("Laplacian root\n(max information density,\nzero tractability)",
                xy=(0, 0), xytext=root_xytext, fontsize=9.5, va="top",
                ha="left", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=1))
    ax.annotate("maximally compressed\n(a single number,\nzero tractability)",
                xy=(0, H), xytext=(1.4, H + 0.4), fontsize=9.5, va="bottom",
                ha="left", color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=1))
    ax.annotate("peak tractability", xy=APEX, xytext=(W + 0.2, H / 2),
                fontsize=9.5, va="center", ha="left", color="#1f5b39")

    ax.set_xlim(-1.8, W + 2.6)
    ax.set_ylim(-3.1 if x_sublabel else -2.6, H + 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


base = "assets/images/representations-entropy-tree_tractability-triangle"
build(base + ".png")
build(base + "-human.png", wedge_label="human\nunderstanding")
build(base + "-machine.png", wedge_label="machine\nunderstanding",
      wedge_color="blue", x_label="machine tractability",
      x_sublabel="(learned readout efficiency)")

# Human vs machine *representations* on the human-tractability axes.
# Machine representations cannot leave the triangle (the boundary is the
# definitional edge of the plane) — they hug the low-tractability hypotenuse,
# the opposite pole from where human representations sit.
build(base + "-representations.png",
      wedge_label="human\nrepresentations", wedge_label_pos=(3.6, 5.0),
      ellipse_region=dict(center=(1.0, 4.0), width=1.3, height=5.3,
                          label="machine representations", color="blue",
                          label_pos=(1.0, 4.0), label_rotation=90),
      repr_label_pos=(1.3, 7.5), repr_label_fontsize=11)
