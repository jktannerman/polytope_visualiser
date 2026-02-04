"""Matplotlib rendering logic for wireframe display."""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

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

    def update(
        self,
        vertices_2d: NDArray[np.float64],
        edges: list[tuple[int, int]],
    ) -> None:
        """Update the wireframe display with new vertex positions.

        Args:
            vertices_2d: Array of shape (N, 2) with projected 2D coordinates.
            edges: List of tuples (i, j) indicating which vertices to connect.
        """
        # Remove existing lines
        for line in self.lines:
            line.remove()
        self.lines.clear()

        # Draw new edges
        for i, j in edges:
            x_coords = [vertices_2d[i, 0], vertices_2d[j, 0]]
            y_coords = [vertices_2d[i, 1], vertices_2d[j, 1]]
            (line,) = self.ax.plot(x_coords, y_coords, color="#00BFFF", linewidth=1.5)
            self.lines.append(line)

        self.ax.figure.canvas.draw_idle()
