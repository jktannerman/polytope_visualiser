"""Matplotlib rendering logic for wireframe display."""

import colorsys
from typing import Any

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

_CYAN_HUE = 195.0 / 360.0  # H of #00BFFF
_W_HUE_SHIFT = 0.10         # max hue displacement (0-1 scale, 0.20 ≈ 72°)

# Axis colours: X=red, Y=green, Z=blue, W=yellow
_AXIS_COLORS_3D = ["#FF4444", "#44FF44", "#4488FF"]
_AXIS_COLORS_4D = ["#FF4444", "#44FF44", "#4488FF", "#FFFF44"]
_AXIS_LABELS = ["X", "Y", "Z", "W"]


class AxesIndicator:
    """Draws a miniature axes orientation indicator with coloured arrows.

    Attributes:
        ax: The Matplotlib axes to draw on.
        artists: List of artists (annotations and texts) to manage.
    """

    def __init__(self, ax: Axes) -> None:
        """Initialize the axes indicator.

        Args:
            ax: Matplotlib axes for drawing the indicator.
        """
        self.ax = ax
        self.artists: list[Any] = []

    def update(
        self,
        basis_2d: NDArray[np.float64],
        depths: NDArray[np.float64],
        dim: int,
    ) -> None:
        """Redraw the axes indicator arrows and labels.

        Args:
            basis_2d: Array of shape (dim, 2) — each row is the 2D projected
                tip of a standard basis vector after rotation.
            depths: 1D array of length dim — the Z component of each rotated
                basis vector (before 2D projection), used for depth cueing.
            dim: Number of axes to draw (3 or 4).
        """
        # Remove old artists
        for artist in self.artists:
            artist.remove()
        self.artists.clear()

        colors = _AXIS_COLORS_4D if dim == 4 else _AXIS_COLORS_3D

        for i in range(dim):
            tip_x, tip_y = basis_2d[i, 0], basis_2d[i, 1]

            # Depth cueing: opacity and linestyle based on Z depth
            alpha = float(np.clip(0.25 + 0.75 * (depths[i] + 1) / 2, 0.25, 1.0))
            linestyle = "--" if depths[i] < 0 else "-"

            # Draw arrow from origin to tip
            ann = self.ax.annotate(
                "",
                xy=(tip_x, tip_y),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colors[i],
                    lw=2,
                    mutation_scale=12,
                    linestyle=linestyle,
                ),
                alpha=alpha,
            )
            self.artists.append(ann)

            # Draw label just past the arrow tip
            label = self.ax.text(
                tip_x * 1.3,
                tip_y * 1.3,
                _AXIS_LABELS[i],
                color=colors[i],
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                alpha=alpha,
            )
            self.artists.append(label)

        self.ax.figure.canvas.draw_idle()


class Renderer:
    """Renders a wireframe polytope on Matplotlib axes.

    Attributes:
        ax: The Matplotlib axes to draw on.
        lines: List of Line2D objects representing edges.
    """

    def __init__(self, ax: Axes) -> None:
        """Initialize the renderer with given axes.

        Args:
            ax: Matplotlib axes for drawing the wireframe.
        """
        self.ax = ax
        self.lines: list[Line2D] = []
        self._collection: LineCollection | None = None

    def update(
        self,
        vertices_2d: NDArray[np.float64],
        edges: list[tuple[int, int]],
        vertex_depths: NDArray[np.float64] | None = None,
        vertex_w_depths: NDArray[np.float64] | None = None,
    ) -> None:
        """Update the wireframe display with new vertex positions.

        Args:
            vertices_2d: Array of shape (N, 2) with projected 2D coordinates.
            edges: List of tuples (i, j) indicating which vertices to connect.
            vertex_depths: Optional 1D array of per-vertex Z depths. When
                provided, edges are drawn with opacity based on average
                endpoint depth (closer = more opaque).
            vertex_w_depths: Optional 1D array of per-vertex W depths. When
                provided, edges are coloured with a hue gradient based on W
                depth (green at ana/high W, purple at kata/low W).
        """
        # Remove existing artists
        for line in self.lines:
            line.remove()
        self.lines.clear()
        if self._collection is not None:
            self._collection.remove()
            self._collection = None

        # Precompute depth-to-alpha mapping
        if vertex_depths is not None:
            z_min = float(vertex_depths.min())
            z_max = float(vertex_depths.max())
            z_range = z_max - z_min
        else:
            z_min = z_max = z_range = 0.0

        if vertex_w_depths is not None:
            # W-depth hue gradient path using LineCollection
            w_min = float(vertex_w_depths.min())
            w_max = float(vertex_w_depths.max())
            w_range = w_max - w_min

            n_sub = 16
            segments: list[list[tuple[float, float]]] = []
            colors: list[tuple[float, float, float, float]] = []

            for i, j in edges:
                p0 = vertices_2d[i]
                p1 = vertices_2d[j]
                w0 = float(vertex_w_depths[i])
                w1 = float(vertex_w_depths[j])

                # Z-depth alpha per vertex
                if vertex_depths is not None and z_range > 0:
                    z_t_i = (vertex_depths[i] - z_min) / z_range
                    z_t_j = (vertex_depths[j] - z_min) / z_range
                    alpha_i = 0.3 + 0.7 * float(z_t_i)
                    alpha_j = 0.3 + 0.7 * float(z_t_j)
                else:
                    alpha_i = 1.0
                    alpha_j = 1.0

                for k in range(n_sub):
                    frac0 = k / n_sub
                    frac1 = (k + 1) / n_sub
                    frac_mid = (frac0 + frac1) / 2

                    seg_p0 = p0 + frac0 * (p1 - p0)
                    seg_p1 = p0 + frac1 * (p1 - p0)
                    segments.append(
                        [(float(seg_p0[0]), float(seg_p0[1])),
                         (float(seg_p1[0]), float(seg_p1[1]))]
                    )

                    # Interpolate W at midpoint
                    w_mid = w0 + frac_mid * (w1 - w0)
                    if w_range > 0:
                        t_signed = 2.0 * (w_mid - w_min) / w_range - 1.0
                    else:
                        t_signed = 0.0

                    hue = (_CYAN_HUE - _W_HUE_SHIFT * t_signed) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

                    # Interpolate alpha at midpoint
                    alpha = alpha_i + frac_mid * (alpha_j - alpha_i)
                    colors.append((r, g, b, alpha))

            lc = LineCollection(segments, colors=colors, linewidths=1.5)
            self.ax.add_collection(lc)
            self._collection = lc
        else:
            # Standard uniform-colour path
            for i, j in edges:
                x_coords = [vertices_2d[i, 0], vertices_2d[j, 0]]
                y_coords = [vertices_2d[i, 1], vertices_2d[j, 1]]

                alpha = 1.0
                if vertex_depths is not None and z_range > 0:
                    avg_depth = (vertex_depths[i] + vertex_depths[j]) / 2
                    t = (avg_depth - z_min) / z_range
                    alpha = 0.3 + 0.7 * t

                (line,) = self.ax.plot(
                    x_coords, y_coords, color="#00BFFF", linewidth=1.5, alpha=alpha
                )
                self.lines.append(line)

        self.ax.figure.canvas.draw_idle()
