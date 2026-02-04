"""Main entry point for the Polytope Visualiser application."""

import ctypes
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from polytopes import (
    create_120cell,
    create_16cell,
    create_24cell,
    create_5cell,
    create_600cell,
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
    max_projected_radius,
    orthogonal_project,
    orthogonal_project_4d,
    perspective_project,
    perspective_project_4d,
)
from renderer import AxesIndicator, Renderer

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

    # Configure checkbutton style
    style.configure(
        "TCheckbutton",
        background=BG_COLOR,
        foreground=FG_COLOR,
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
        "5-cell": create_5cell,
        "Tesseract": create_tesseract,
        "16-cell": create_16cell,
        "24-cell": create_24cell,
        "120-cell": create_120cell,
        "600-cell": create_600cell,
    }

    # Create the initial polytope (use list for mutability in nested functions)
    current_polytope = [create_cube()]

    # Create main Tkinter window
    root = tk.Tk()
    root.title("Wireframe Visualizer")
    root.geometry("750x700")
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

    # Depth opacity checkbox
    depth_opacity_var = tk.BooleanVar(value=False)
    depth_opacity_check = ttk.Checkbutton(
        top_frame,
        text="Depth opacity",
        variable=depth_opacity_var,
        command=lambda: update(),
    )
    depth_opacity_check.pack(side=tk.LEFT, padx=(0, 20))

    # W-depth hue checkbox (visible only for 4D polytopes)
    w_hue_var = tk.BooleanVar(value=False)
    w_hue_check = ttk.Checkbutton(
        top_frame,
        text="W-depth hue",
        variable=w_hue_var,
        command=lambda: update(),
    )

    # Distance slider (for perspective projection)
    distance_label = ttk.Label(top_frame, text="Distance:")
    distance_var = tk.DoubleVar(value=4.0)
    distance_slider = ttk.Scale(
        top_frame,
        from_=1.5,
        to=10.0,
        orient=tk.HORIZONTAL,
        variable=distance_var,
        length=100,
    )
    distance_entry = tk.Entry(
        top_frame, width=5, justify=tk.RIGHT,
        bg=ENTRY_BG, fg=FG_COLOR, insertbackground=FG_COLOR,
        relief=tk.FLAT, highlightthickness=1, highlightcolor="#555555",
        highlightbackground=ACCENT_COLOR,
    )
    distance_entry.insert(0, "4.0")

    def _commit_distance(_: tk.Event | None = None) -> None:
        text = distance_entry.get().strip()
        try:
            val = max(1.5, min(10.0, float(text)))
        except ValueError:
            val = distance_var.get()
        distance_var.set(val)
        distance_entry.delete(0, tk.END)
        distance_entry.insert(0, f"{val:.1f}")

    distance_entry.bind("<Return>", _commit_distance)
    distance_entry.bind("<FocusOut>", _commit_distance)
    distance_entry.bind("<FocusIn>", lambda _: distance_entry.select_range(0, tk.END))
    distance_entry.bind("<Escape>", lambda _: root.focus_set())

    def update_distance_visibility() -> None:
        """Show/hide distance slider based on projection type."""
        if projection_var.get() == "Perspective":
            distance_label.pack(side=tk.LEFT, padx=(0, 5))
            distance_slider.pack(side=tk.LEFT, padx=(0, 5))
            distance_entry.pack(side=tk.LEFT)
        else:
            distance_label.pack_forget()
            distance_slider.pack_forget()
            distance_entry.pack_forget()

    # Initially hide distance controls (Orthogonal is default)
    update_distance_visibility()

    # Create matplotlib figure with dark background
    fig = Figure(figsize=(7.5, 6), dpi=100, facecolor=BG_COLOR)

    # Main polytope view — full figure, centered
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    # Axes indicator overlay at left-center
    ax_indicator = fig.add_axes([0.0, 0.35, 0.15, 0.3])
    ax_indicator.set_facecolor("none")
    ax_indicator.set_aspect("equal")
    ax_indicator.axis("off")
    ax_indicator.set_xlim(-1.5, 1.5)
    ax_indicator.set_ylim(-1.5, 1.5)

    # Create canvas and embed in window
    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.configure(bg=BG_COLOR, highlightthickness=0)
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Create renderer and axes indicator
    renderer = Renderer(ax)
    axes_indicator = AxesIndicator(ax_indicator)

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
    slider_widgets: list[tuple[ttk.Label, ttk.Scale, tk.Entry]] = []

    def update(_: str | None = None) -> None:
        """Update the visualization when sliders change."""
        polytope = current_polytope[0]
        use_perspective = projection_var.get() == "Perspective"
        distance = distance_var.get()

        # Update distance entry (only when not being edited)
        if root.focus_get() != distance_entry:
            distance_entry.delete(0, tk.END)
            distance_entry.insert(0, f"{distance:.1f}")

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

            # Vertex depths for edge opacity (Z of 3D intermediate)
            vertex_depths = rotated_3d[:, 2]

            # W depths for hue gradient (W of rotated 4D vertices)
            w_depths = rotated_4d[:, 3]

            # Axes indicator: always orthogonal projection of rotated basis
            rotated_basis_4d = apply_rotation_4d(
                np.eye(4), rxy, rxz, rxw, ryz, ryw, rzw
            )
            basis_3d = orthogonal_project_4d(rotated_basis_4d)
            depths = basis_3d[:, 2]
            basis_2d = orthogonal_project(basis_3d)
            axes_indicator.update(basis_2d, depths, dim=4)
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

            # Vertex depths for edge opacity (Z of rotated 3D vertices)
            vertex_depths = rotated[:, 2]

            # Axes indicator: always orthogonal projection of rotated basis
            rotated_basis_3d = apply_rotation(np.eye(3), rx, ry, rz)
            depths = rotated_basis_3d[:, 2]
            basis_2d = orthogonal_project(rotated_basis_3d)
            axes_indicator.update(basis_2d, depths, dim=3)

        # Set axis limits analytically based on projection bounds
        r_max = max_projected_radius(
            projection_var.get(), polytope.dim, distance
        )
        margin = r_max * 1.05
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)

        # Update display
        renderer.update(
            projected,
            polytope.edges,
            vertex_depths if depth_opacity_var.get() else None,
            w_depths if (polytope.dim == 4 and w_hue_var.get()) else None,
        )
        canvas.draw_idle()

    def update_slider_visibility() -> None:
        """Show/hide sliders and 4D controls based on current polytope dimension."""
        polytope = current_polytope[0]
        is_4d = polytope.dim == 4

        for i, (label_widget, slider_widget, value_entry) in enumerate(slider_widgets):
            if i < 3 or is_4d:
                # Show first 3 always, show all 6 for 4D
                label_widget.grid()
                slider_widget.grid()
                value_entry.grid()
            else:
                # Hide extra sliders for 3D
                label_widget.grid_remove()
                slider_widget.grid_remove()
                value_entry.grid_remove()

        # Show/hide W-depth hue checkbox
        if is_4d:
            w_hue_check.pack(side=tk.LEFT, padx=(0, 20))
        else:
            w_hue_check.pack_forget()

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
    ) -> tuple[ttk.Label, ttk.Scale, tk.Entry]:
        """Create a labeled slider row with an editable value entry."""
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

        value_entry = tk.Entry(
            parent, width=5, justify=tk.RIGHT,
            bg=ENTRY_BG, fg=FG_COLOR, insertbackground=FG_COLOR,
            relief=tk.FLAT, highlightthickness=1, highlightcolor="#555555",
            highlightbackground=ACCENT_COLOR,
        )
        value_entry.grid(row=row, column=2, sticky=tk.E, pady=5)
        value_entry.insert(0, "0°")

        def sync_entry(*_: object) -> None:
            if root.focus_get() != value_entry:
                value_entry.delete(0, tk.END)
                value_entry.insert(0, f"{int(variable.get())}°")

        def commit_entry(_: tk.Event | None = None) -> None:
            text = value_entry.get().strip().rstrip("°").strip()
            try:
                val = max(0.0, min(360.0, float(text)))
            except ValueError:
                val = variable.get()
            variable.set(val)
            value_entry.delete(0, tk.END)
            value_entry.insert(0, f"{int(variable.get())}°")
            update()

        variable.trace_add("write", sync_entry)
        value_entry.bind("<Return>", commit_entry)
        value_entry.bind("<FocusOut>", commit_entry)
        value_entry.bind("<FocusIn>", lambda _: value_entry.select_range(0, tk.END))
        value_entry.bind("<Escape>", lambda _: root.focus_set())
        return label_widget, slider, value_entry

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
