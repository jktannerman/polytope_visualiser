"""Main entry point for the Polytope Visualiser application."""

import ctypes
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from polytopes import create_cube
from transforms import apply_rotation, orthogonal_project
from renderer import Renderer

# Dark mode colors
BG_COLOR = "#1a1a1a"
FG_COLOR = "#e0e0e0"
ACCENT_COLOR = "#2d2d2d"
ENTRY_BG = "#333333"


def enable_dark_title_bar(window: tk.Tk) -> None:
    """Enable dark title bar on Windows 10/11."""
    if sys.platform != "win32":
        return

    window.update()
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
    value = ctypes.c_int(1)
    ctypes.windll.dwmapi.DwmSetWindowAttribute(
        hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value)
    )


def configure_dark_style() -> ttk.Style:
    """Configure ttk styles for dark mode."""
    style = ttk.Style()
    style.theme_use("clam")

    # Configure frame style
    style.configure("TFrame", background=BG_COLOR)

    # Configure label style
    style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR)

    # Configure combobox style
    style.configure(
        "TCombobox",
        fieldbackground=ENTRY_BG,
        background=ACCENT_COLOR,
        foreground=FG_COLOR,
        arrowcolor=FG_COLOR,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", ENTRY_BG)],
        selectbackground=[("readonly", ENTRY_BG)],
        selectforeground=[("readonly", FG_COLOR)],
    )

    # Configure scale (slider) style
    style.configure(
        "TScale",
        background=BG_COLOR,
        troughcolor=ACCENT_COLOR,
        sliderthickness=15,
    )

    return style


def main() -> None:
    """Run the polytope visualizer application."""
    # Create the polytope
    polytope = create_cube()

    # Create main Tkinter window
    root = tk.Tk()
    root.title(f"{polytope.name} - Wireframe Visualizer")
    root.geometry("600x700")
    root.configure(bg=BG_COLOR)

    # Enable dark title bar on Windows
    enable_dark_title_bar(root)

    # Configure dark styles
    configure_dark_style()

    # Top frame for future dropdowns (placeholder)
    top_frame = ttk.Frame(root, padding="10")
    top_frame.pack(fill=tk.X)

    # Placeholder labels for future dropdowns
    ttk.Label(top_frame, text="Polytope:").pack(side=tk.LEFT, padx=(0, 5))
    polytope_combo = ttk.Combobox(top_frame, values=["Cube"], state="readonly", width=15)
    polytope_combo.set("Cube")
    polytope_combo.pack(side=tk.LEFT, padx=(0, 20))

    ttk.Label(top_frame, text="Projection:").pack(side=tk.LEFT, padx=(0, 5))
    projection_combo = ttk.Combobox(
        top_frame, values=["Orthogonal"], state="readonly", width=15
    )
    projection_combo.set("Orthogonal")
    projection_combo.pack(side=tk.LEFT)

    # Create matplotlib figure with dark background
    fig = Figure(figsize=(6, 6), dpi=100, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    # Configure axes - no grid, no ticks, no axes lines
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Create canvas and embed in window
    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.configure(bg=BG_COLOR, highlightthickness=0)
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Create renderer
    renderer = Renderer(ax)

    # Bottom frame for sliders
    slider_frame = ttk.Frame(root, padding="10")
    slider_frame.pack(fill=tk.X, side=tk.BOTTOM)

    # Variables to store slider values
    rx_var = tk.DoubleVar(value=0)
    ry_var = tk.DoubleVar(value=0)
    rz_var = tk.DoubleVar(value=0)

    def update(_: str | None = None) -> None:
        """Update the visualization when sliders change."""
        # Get angles in radians
        rx = np.radians(rx_var.get())
        ry = np.radians(ry_var.get())
        rz = np.radians(rz_var.get())

        # Apply rotation and projection
        rotated = apply_rotation(polytope.vertices, rx, ry, rz)
        projected = orthogonal_project(rotated)

        # Update display
        renderer.update(projected, polytope.edges)
        canvas.draw_idle()

    # Create slider rows
    def create_slider_row(
        parent: ttk.Frame, label: str, variable: tk.DoubleVar, row: int
    ) -> ttk.Scale:
        """Create a labeled slider row."""
        ttk.Label(parent, text=f"{label}:", width=12).grid(
            row=row, column=0, sticky=tk.W, pady=5
        )
        slider = ttk.Scale(
            parent,
            from_=0,
            to=360,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=update,
        )
        slider.grid(row=row, column=1, sticky=tk.EW, padx=(5, 10), pady=5)

        value_label = ttk.Label(parent, text="0°", width=5)
        value_label.grid(row=row, column=2, sticky=tk.E, pady=5)

        def update_label(_: str | None = None) -> None:
            value_label.config(text=f"{int(variable.get())}°")

        variable.trace_add("write", lambda *_: update_label())
        return slider

    # Configure grid columns
    slider_frame.columnconfigure(1, weight=1)

    # Create the three rotation sliders
    create_slider_row(slider_frame, "X Rotation", rx_var, 0)
    create_slider_row(slider_frame, "Y Rotation", ry_var, 1)
    create_slider_row(slider_frame, "Z Rotation", rz_var, 2)

    # Initial render
    update()

    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
