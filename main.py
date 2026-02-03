"""Main entry point for the Polytope Visualiser application."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from polytopes import create_cube
from transforms import apply_rotation, orthogonal_project
from renderer import Renderer


def main() -> None:
    """Run the polytope visualizer application."""
    # Create the polytope
    polytope = create_cube()

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.3)

    # Configure main axes
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title(f"{polytope.name} - Wireframe Visualizer")
    ax.grid(True, alpha=0.3)

    # Create renderer
    renderer = Renderer(ax)

    # Create slider axes
    ax_rx = plt.axes([0.2, 0.2, 0.6, 0.03])
    ax_ry = plt.axes([0.2, 0.13, 0.6, 0.03])
    ax_rz = plt.axes([0.2, 0.06, 0.6, 0.03])

    # Create sliders (display in degrees, store in degrees)
    slider_rx = Slider(ax_rx, "X Rotation", 0, 360, valinit=0, valstep=1)
    slider_ry = Slider(ax_ry, "Y Rotation", 0, 360, valinit=0, valstep=1)
    slider_rz = Slider(ax_rz, "Z Rotation", 0, 360, valinit=0, valstep=1)

    def update(_: float | None = None) -> None:
        """Update the visualization when sliders change."""
        # Get angles in radians
        rx = np.radians(slider_rx.val)
        ry = np.radians(slider_ry.val)
        rz = np.radians(slider_rz.val)

        # Apply rotation and projection
        rotated = apply_rotation(polytope.vertices, rx, ry, rz)
        projected = orthogonal_project(rotated)

        # Update display
        renderer.update(projected, polytope.edges)

    # Connect sliders to update function
    slider_rx.on_changed(update)
    slider_ry.on_changed(update)
    slider_rz.on_changed(update)

    # Initial render
    update()

    plt.show()


if __name__ == "__main__":
    main()
