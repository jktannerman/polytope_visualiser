# Polytope Visualiser

A wireframe visualiser for 3D and 4D polytopes with interactive rotation controls.

## Features

- **3D Polytopes**: All 5 Platonic solids (Tetrahedron, Cube, Octahedron, Icosahedron, Dodecahedron)
- **4D Polytopes**: 5-cell, Tesseract, 16-cell
- **Projection Modes**: Orthogonal and perspective projection
- **Interactive Rotation**: Real-time rotation via sliders
- **Dynamic UI**: Automatically shows 3 sliders for 3D shapes, 6 for 4D
- **Depth-Cued Axes Indicator**: Axis arrows fade and become dashed when pointing away from the viewer
- **Dark Mode**: Native dark theme with dark title bar on Windows

## Running

```powershell
py -3.13 polytope_visualiser/main.py
```

## Requirements

- Python 3.13
- NumPy
- Matplotlib

## How It Works

### Rotation

**3D polytopes** rotate around 3 axes (X, Y, Z), equivalent to rotation in 3 planes (YZ, XZ, XY).

**4D polytopes** rotate in 6 planes:

| Plane | Coordinates Affected |
|-------|---------------------|
| XY | X, Y |
| XZ | X, Z |
| XW | X, W |
| YZ | Y, Z |
| YW | Y, W |
| ZW | Z, W |

### Projection Pipeline

**Orthogonal projection** (default):
```
4D vertices → [4D rotation] → [Drop W] → 3D → [Drop Z] → 2D screen
```

**Perspective projection**: Applies depth-based scaling before dropping coordinates, creating a sense of depth where closer vertices appear larger.

### Axis Limits and Projection Bounds

The 2D axis limits are set **analytically** rather than by rescaling vertex coordinates on the fly. All polytope vertices lie on a unit hypersphere (radius 1), so the maximum projected radius depends only on the projection type, dimension, and camera distance — not on rotation.

| Pipeline | Max projected radius | Constraint |
|----------|---------------------|------------|
| Orthogonal (3D or 4D) | `1.0` | None |
| 3D perspective | `d / sqrt(d² - 1)` | `d > 1` |
| 4D double perspective | `d / sqrt(d² - 2)` | `d > √2` |

The `max_projected_radius()` function in `transforms.py` computes these bounds. The result is applied each frame with a 5% margin (`r_max * 1.05`) to set `ax.set_xlim` / `ax.set_ylim`. This ensures:
- No vertex ever clips outside the canvas
- The polytope fills the canvas (no large empty margins)
- Changing distance smoothly rescales the axes without jumps
- Rotation never changes the apparent size of the polytope

The distance slider minimum is `1.5`, chosen to stay safely above the `√2 ≈ 1.414` constraint for 4D double perspective.

### Axes Indicator Depth Cueing

The axes indicator overlay uses depth cues to convey which axes point towards or away from the viewer:

- **Opacity**: The Z component of each rotated basis vector is linearly mapped from `[-1, +1]` to alpha `[0.25, 1.0]`. Axes pointing towards the viewer are fully opaque; axes pointing away are semi-transparent.
- **Line style**: Axes with negative Z (pointing away) are drawn with dashed lines; axes with non-negative Z (pointing towards) use solid lines.

Both cues are applied to the arrow and its label.

### Rotation Order

The 4D rotation order matters. Since projection discards Z and W, rotations that only affect Z/W (like ZW) must be applied early so subsequent rotations can propagate their effects into X/Y.

## Project Structure

```
polytope_visualiser/
├── main.py        # Application entry point and UI
├── polytopes.py   # Polytope definitions (vertices, edges)
├── transforms.py  # Rotation matrices and projections
└── renderer.py    # Matplotlib rendering
```
