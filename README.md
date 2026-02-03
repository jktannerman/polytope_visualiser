# Polytope Visualiser

A wireframe visualiser for 3D and 4D polytopes with interactive rotation controls.

## Features

- **3D Polytopes**: All 5 Platonic solids (Tetrahedron, Cube, Octahedron, Icosahedron, Dodecahedron)
- **4D Polytopes**: 5-cell, Tesseract, 16-cell
- **Projection Modes**: Orthogonal and perspective projection
- **Interactive Rotation**: Real-time rotation via sliders
- **Dynamic UI**: Automatically shows 3 sliders for 3D shapes, 6 for 4D
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
