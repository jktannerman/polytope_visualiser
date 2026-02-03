"""Polytope definitions with vertices and edges."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Polytope:
    """A polytope defined by vertices and edges.

    Attributes:
        vertices: Array of shape (N, D) containing D-dimensional vertex coordinates.
        edges: List of tuples (i, j) indicating which vertices to connect.
        name: Human-readable name for the polytope.
    """

    vertices: NDArray[np.float64]
    edges: list[tuple[int, int]]
    name: str = "Polytope"

    @property
    def dim(self) -> int:
        """Return the dimension of the polytope (3 for 3D, 4 for 4D, etc.)."""
        return self.vertices.shape[1]


def create_cube() -> Polytope:
    """Create a unit cube centered at the origin.

    Returns:
        Polytope with 8 vertices at (±1, ±1, ±1) and 12 edges.
    """
    vertices = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )

    # Edges connect vertices that differ in exactly one coordinate
    edges = [
        # Bottom face (z = -1)
        (0, 2),
        (2, 6),
        (6, 4),
        (4, 0),
        # Top face (z = 1)
        (1, 3),
        (3, 7),
        (7, 5),
        (5, 1),
        # Vertical edges
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]

    return Polytope(vertices=vertices, edges=edges, name="Cube")


def create_octahedron() -> Polytope:
    """Create a regular octahedron centered at the origin.

    Returns:
        Polytope with 6 vertices at (±1, 0, 0), (0, ±1, 0), (0, 0, ±1) and 12 edges.
    """
    vertices = np.array(
        [
            [1, 0, 0],   # 0: +X
            [-1, 0, 0],  # 1: -X
            [0, 1, 0],   # 2: +Y
            [0, -1, 0],  # 3: -Y
            [0, 0, 1],   # 4: +Z
            [0, 0, -1],  # 5: -Z
        ],
        dtype=np.float64,
    )

    # Each vertex connects to 4 others (all except its opposite)
    edges = [
        # Edges from +X vertex
        (0, 2), (0, 3), (0, 4), (0, 5),
        # Edges from -X vertex
        (1, 2), (1, 3), (1, 4), (1, 5),
        # Remaining edges (connecting Y and Z vertices)
        (2, 4), (2, 5), (3, 4), (3, 5),
    ]

    return Polytope(vertices=vertices, edges=edges, name="Octahedron")


def create_tesseract() -> Polytope:
    """Create a tesseract (4D hypercube) centered at the origin.

    Returns:
        Polytope with 16 vertices at (±1, ±1, ±1, ±1) and 32 edges.
    """
    # Generate all 16 vertices: all combinations of (±1, ±1, ±1, ±1)
    vertices = np.array(
        [
            [x, y, z, w]
            for x in [-1, 1]
            for y in [-1, 1]
            for z in [-1, 1]
            for w in [-1, 1]
        ],
        dtype=np.float64,
    )

    # Edges connect vertices that differ in exactly one coordinate
    edges: list[tuple[int, int]] = []
    for i in range(16):
        for j in range(i + 1, 16):
            # Count how many coordinates differ
            diff_count = np.sum(vertices[i] != vertices[j])
            if diff_count == 1:
                edges.append((i, j))

    return Polytope(vertices=vertices, edges=edges, name="Tesseract")
