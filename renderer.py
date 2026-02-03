"""Matplotlib rendering logic for wireframe display."""

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray


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
            (line,) = self.ax.plot(x_coords, y_coords, "b-", linewidth=1.5)
            self.lines.append(line)

        self.ax.figure.canvas.draw_idle()
