"""Main entry point for the Polytope Visualiser application."""

import ctypes
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from polytopes import (
    create_cube,
    create_dodecahedron,
    create_icosahedron,
    create_octahedron,
    create_tetrahedron,
    create_tesseract,
)
from transforms import (
    apply_rotation,
    apply_rotation_4d,
    orthogonal_project,
    orthogonal_project_4d,
    perspective_project,
    perspective_project_4d,
)
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
    # Polytope factory functions
    polytope_factories = {
        "Tetrahedron": create_tetrahedron,
        "Cube": create_cube,
        "Octahedron": create_octahedron,
        "Icosahedron": create_icosahedron,
        "Dodecahedron": create_dodecahedron,
        "Tesseract": create_tesseract,
    }

    # Create the initial polytope (use list for mutability in nested functions)
    current_polytope = [create_cube()]

    # Create main Tkinter window
    root = tk.Tk()
    root.title("Wireframe Visualizer")
    root.geometry("600x700")
    root.configure(bg=BG_COLOR)

    # Enable dark title bar on Windows
    enable_dark_title_bar(root)

    # Configure dark styles
    configure_dark_style()

    # Top frame for future dropdowns (placeholder)
    top_frame = ttk.Frame(root, padding="10")
    top_frame.pack(fill=tk.X)

    # Polytope dropdown
    ttk.Label(top_frame, text="Polytope:").pack(side=tk.LEFT, padx=(0, 5))
    polytope_names = list(polytope_factories.keys())
    polytope_var = tk.StringVar(value="Cube")
    polytope_combo = ttk.Combobox(
        top_frame,
        values=polytope_names,
        state="readonly",
        width=15,
        textvariable=polytope_var,
    )
    polytope_combo.pack(side=tk.LEFT, padx=(0, 20))

    ttk.Label(top_frame, text="Projection:").pack(side=tk.LEFT, padx=(0, 5))
    projection_var = tk.StringVar(value="Orthogonal")
    projection_combo = ttk.Combobox(
        top_frame,
        values=["Orthogonal", "Perspective"],
        state="readonly",
        width=15,
        textvariable=projection_var,
    )
    projection_combo.pack(side=tk.LEFT, padx=(0, 20))

    # Distance slider (for perspective projection)
    distance_label = ttk.Label(top_frame, text="Distance:")
    distance_var = tk.DoubleVar(value=4.0)
    distance_slider = ttk.Scale(
        top_frame,
        from_=2.5,
        to=10.0,
        orient=tk.HORIZONTAL,
        variable=distance_var,
        length=100,
    )
    distance_value_label = ttk.Label(top_frame, text="4.0", width=4)

    def update_distance_visibility() -> None:
        """Show/hide distance slider based on projection type."""
        if projection_var.get() == "Perspective":
            distance_label.pack(side=tk.LEFT, padx=(0, 5))
            distance_slider.pack(side=tk.LEFT, padx=(0, 5))
            distance_value_label.pack(side=tk.LEFT)
        else:
            distance_label.pack_forget()
            distance_slider.pack_forget()
            distance_value_label.pack_forget()

    # Initially hide distance controls (Orthogonal is default)
    update_distance_visibility()

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

    # Slider configuration: (label, variable_name) for 3D and 4D
    # 3D uses first 3 (XY, XZ, YZ correspond to Z, Y, X axis rotations)
    # 4D uses all 6
    slider_configs = [
        ("XY Rotation", "rxy"),  # 3D: equivalent to Z-axis rotation
        ("XZ Rotation", "rxz"),  # 3D: equivalent to Y-axis rotation
        ("YZ Rotation", "ryz"),  # 3D: equivalent to X-axis rotation
        ("XW Rotation", "rxw"),  # 4D only
        ("YW Rotation", "ryw"),  # 4D only
        ("ZW Rotation", "rzw"),  # 4D only
    ]

    # Variables to store slider values (all 6)
    slider_vars: dict[str, tk.DoubleVar] = {}
    for _, var_name in slider_configs:
        slider_vars[var_name] = tk.DoubleVar(value=0)

    # Storage for slider widgets (for show/hide)
    slider_widgets: list[tuple[ttk.Label, ttk.Scale, ttk.Label]] = []

    def update(_: str | None = None) -> None:
        """Update the visualization when sliders change."""
        polytope = current_polytope[0]
        use_perspective = projection_var.get() == "Perspective"
        distance = distance_var.get()

        # Update distance value label
        distance_value_label.config(text=f"{distance:.1f}")

        if polytope.dim == 4:
            # 4D rotation and projection
            rxy = np.radians(slider_vars["rxy"].get())
            rxz = np.radians(slider_vars["rxz"].get())
            rxw = np.radians(slider_vars["rxw"].get())
            ryz = np.radians(slider_vars["ryz"].get())
            ryw = np.radians(slider_vars["ryw"].get())
            rzw = np.radians(slider_vars["rzw"].get())

            rotated_4d = apply_rotation_4d(
                polytope.vertices, rxy, rxz, rxw, ryz, ryw, rzw
            )

            if use_perspective:
                # Perspective: 4D→3D then 3D→2D
                rotated_3d = perspective_project_4d(rotated_4d, distance)
                projected = perspective_project(rotated_3d, distance)
            else:
                # Orthogonal
                rotated_3d = orthogonal_project_4d(rotated_4d)
                projected = orthogonal_project(rotated_3d)
        else:
            # 3D rotation and projection
            # Map XY/XZ/YZ plane rotations to X/Y/Z axis rotations
            rx = np.radians(slider_vars["ryz"].get())  # YZ plane = X axis
            ry = np.radians(slider_vars["rxz"].get())  # XZ plane = Y axis
            rz = np.radians(slider_vars["rxy"].get())  # XY plane = Z axis

            rotated = apply_rotation(polytope.vertices, rx, ry, rz)

            if use_perspective:
                projected = perspective_project(rotated, distance)
            else:
                projected = orthogonal_project(rotated)

        # Normalize to fit within view bounds (max ~1.8 to leave margin)
        max_coord = np.abs(projected).max()
        if max_coord > 1.8:
            projected = projected * (1.8 / max_coord)

        # Update display
        renderer.update(projected, polytope.edges)
        canvas.draw_idle()

    def update_slider_visibility() -> None:
        """Show/hide sliders based on current polytope dimension."""
        polytope = current_polytope[0]
        is_4d = polytope.dim == 4

        for i, (label_widget, slider_widget, value_label) in enumerate(slider_widgets):
            if i < 3 or is_4d:
                # Show first 3 always, show all 6 for 4D
                label_widget.grid()
                slider_widget.grid()
                value_label.grid()
            else:
                # Hide extra sliders for 3D
                label_widget.grid_remove()
                slider_widget.grid_remove()
                value_label.grid_remove()

    def on_polytope_change(_: tk.Event | None = None) -> None:
        """Handle polytope selection change."""
        name = polytope_var.get()
        current_polytope[0] = polytope_factories[name]()

        # Reset all sliders to 0
        for var in slider_vars.values():
            var.set(0)

        # Update slider visibility
        update_slider_visibility()
        update()

    polytope_combo.bind("<<ComboboxSelected>>", on_polytope_change)

    def on_projection_change(_: tk.Event | None = None) -> None:
        """Handle projection type change."""
        update_distance_visibility()
        update()

    projection_combo.bind("<<ComboboxSelected>>", on_projection_change)

    # Bind distance slider to update
    distance_var.trace_add("write", lambda *_: update())

    # Create slider rows
    def create_slider_row(
        parent: ttk.Frame, label: str, variable: tk.DoubleVar, row: int
    ) -> tuple[ttk.Label, ttk.Scale, ttk.Label]:
        """Create a labeled slider row and return widgets for visibility control."""
        label_widget = ttk.Label(parent, text=f"{label}:", width=12)
        label_widget.grid(row=row, column=0, sticky=tk.W, pady=5)

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
        return label_widget, slider, value_label

    # Configure grid columns
    slider_frame.columnconfigure(1, weight=1)

    # Create all 6 rotation sliders
    for i, (label, var_name) in enumerate(slider_configs):
        widgets = create_slider_row(slider_frame, label, slider_vars[var_name], i)
        slider_widgets.append(widgets)

    # Initialize slider visibility for the current polytope
    update_slider_visibility()

    # Initial render
    update()

    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()
