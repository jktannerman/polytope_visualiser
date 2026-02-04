"""Polytope definitions with vertices and edges."""

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from numpy.typing import NDArray


@dataclass
class Polytope:
    """A polytope defined by vertices and edges.

    All creation functions normalise vertices to lie on the unit hypersphere
    (distance 1 from the origin).  The projection-bounds logic in
    ``transforms.max_projected_radius()`` depends on this invariant.

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

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

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


def create_tetrahedron() -> Polytope:
    """Create a regular tetrahedron centered at the origin.

    Returns:
        Polytope with 4 vertices and 6 edges.
    """
    # Tetrahedron inscribed in a cube: alternating vertices
    vertices = np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=np.float64,
    )

    # Every vertex connects to every other vertex (complete graph K4)
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3),
    ]

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="Tetrahedron")


def create_icosahedron() -> Polytope:
    """Create a regular icosahedron centered at the origin.

    Returns:
        Polytope with 12 vertices and 30 edges.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Vertices are cyclic permutations of (0, ±1, ±φ)
    vertices = np.array(
        [
            [0, 1, phi],
            [0, 1, -phi],
            [0, -1, phi],
            [0, -1, -phi],
            [1, phi, 0],
            [1, -phi, 0],
            [-1, phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, 1],
            [-phi, 0, -1],
        ],
        dtype=np.float64,
    )

    # Edge length is 2, so connect vertices at distance 2
    edge_length_sq = 4.0
    edges: list[tuple[int, int]] = []
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((vertices[i] - vertices[j]) ** 2)
            if np.isclose(dist_sq, edge_length_sq):
                edges.append((i, j))

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="Icosahedron")


def create_dodecahedron() -> Polytope:
    """Create a regular dodecahedron centered at the origin.

    Returns:
        Polytope with 20 vertices and 30 edges.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi  # = φ - 1

    vertices_list = []

    # 8 cube vertices: (±1, ±1, ±1)
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vertices_list.append([x, y, z])

    # 12 vertices from cyclic permutations of (0, ±1/φ, ±φ)
    for sign1 in [-1, 1]:
        for sign2 in [-1, 1]:
            vertices_list.append([0, sign1 * inv_phi, sign2 * phi])
            vertices_list.append([sign1 * inv_phi, sign2 * phi, 0])
            vertices_list.append([sign2 * phi, 0, sign1 * inv_phi])

    vertices = np.array(vertices_list, dtype=np.float64)

    # Edge length squared is (2/φ)² = 4/φ² ≈ 1.528
    edge_length_sq = 4 / (phi * phi)
    edges: list[tuple[int, int]] = []
    for i in range(20):
        for j in range(i + 1, 20):
            dist_sq = np.sum((vertices[i] - vertices[j]) ** 2)
            if np.isclose(dist_sq, edge_length_sq):
                edges.append((i, j))

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="Dodecahedron")


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

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="Tesseract")


def create_5cell() -> Polytope:
    """Create a 5-cell (pentachoron) centered at the origin.

    The 5-cell is the 4D analogue of the tetrahedron, the simplest
    regular 4D polytope.

    Returns:
        Polytope with 5 vertices and 10 edges.
    """
    # 4 vertices form a tetrahedron in the w = -1/√5 hyperplane,
    # with a 5th vertex along the +w axis
    sqrt5 = np.sqrt(5)
    vertices = np.array(
        [
            [1, 1, 1, -1 / sqrt5],
            [1, -1, -1, -1 / sqrt5],
            [-1, 1, -1, -1 / sqrt5],
            [-1, -1, 1, -1 / sqrt5],
            [0, 0, 0, 4 / sqrt5],
        ],
        dtype=np.float64,
    )

    # Every vertex connects to every other vertex (complete graph K5)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4),
    ]

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="5-cell")


def create_16cell() -> Polytope:
    """Create a 16-cell (hexadecachoron) centered at the origin.

    The 16-cell is the 4D analogue of the octahedron and the dual
    of the tesseract.

    Returns:
        Polytope with 8 vertices and 24 edges.
    """
    # Vertices at ±1 along each axis
    vertices = np.array(
        [
            [1, 0, 0, 0],   # 0: +X
            [-1, 0, 0, 0],  # 1: -X
            [0, 1, 0, 0],   # 2: +Y
            [0, -1, 0, 0],  # 3: -Y
            [0, 0, 1, 0],   # 4: +Z
            [0, 0, -1, 0],  # 5: -Z
            [0, 0, 0, 1],   # 6: +W
            [0, 0, 0, -1],  # 7: -W
        ],
        dtype=np.float64,
    )

    # Each vertex connects to all others except its opposite
    # Opposite pairs: (0,1), (2,3), (4,5), (6,7)
    edges: list[tuple[int, int]] = []
    for i in range(8):
        for j in range(i + 1, 8):
            # Skip if they are opposites (same axis)
            if (i // 2) != (j // 2):
                edges.append((i, j))

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="16-cell")


# The 12 even permutations of indices (0, 1, 2, 3)
_EVEN_PERM_INDICES: list[tuple[int, ...]] = [
    (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
    (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
    (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
    (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
]


def _signed_permutations(
    base: tuple[float, ...], *, even_only: bool = False
) -> list[list[float]]:
    """Generate 4D vertices from coordinate permutations with sign changes.

    Args:
        base: Tuple of 4 non-negative coordinate magnitudes.
        even_only: If True, use only the 12 even permutations.

    Returns:
        Deduplicated list of 4D coordinate lists.
    """
    perms = _EVEN_PERM_INDICES if even_only else list(permutations(range(4)))
    seen: set[tuple[float, ...]] = set()
    result: list[list[float]] = []

    for perm in perms:
        permuted = tuple(base[i] for i in perm)
        sign_choices: list[list[float]] = []
        for v in permuted:
            if v == 0.0:
                sign_choices.append([0.0])
            else:
                sign_choices.append([v, -v])
        for combo in product(*sign_choices):
            key = tuple(round(x, 10) for x in combo)
            if key not in seen:
                seen.add(key)
                result.append(list(combo))

    return result


def _edges_by_distance(
    vertices: NDArray[np.float64], edge_length_sq: float
) -> list[tuple[int, int]]:
    """Find edges by identifying vertex pairs at a given squared distance.

    Args:
        vertices: Array of shape (N, D) with vertex coordinates.
        edge_length_sq: Squared edge length to match.

    Returns:
        List of (i, j) index pairs with i < j.
    """
    n = len(vertices)
    diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    i_indices, j_indices = np.where(
        np.isclose(dist_sq, edge_length_sq)
        & (np.arange(n)[:, None] < np.arange(n)[None, :])
    )
    return [(int(i), int(j)) for i, j in zip(i_indices, j_indices)]


def create_24cell() -> Polytope:
    """Create a 24-cell (icositetrachoron) centered at the origin.

    The 24-cell is a self-dual regular 4D polytope with no 3D analogue.

    Returns:
        Polytope with 24 vertices and 96 edges.
    """
    # Vertices: all permutations of (±1, ±1, 0, 0)
    vertices = np.array(
        _signed_permutations((1.0, 1.0, 0.0, 0.0)),
        dtype=np.float64,
    )

    # Edge length equals circumradius: √2
    edge_length_sq = 2.0
    edges = _edges_by_distance(vertices, edge_length_sq)

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="24-cell")


def create_600cell() -> Polytope:
    """Create a 600-cell (hexacosichoron) centered at the origin.

    The 600-cell is the 4D analogue of the icosahedron with 600
    tetrahedral cells.

    Returns:
        Polytope with 120 vertices and 720 edges.
    """
    phi = (1 + np.sqrt(5)) / 2

    vertices_list: list[list[float]] = []

    # Group 1 (8 vertices): permutations of (±2, 0, 0, 0)
    vertices_list.extend(_signed_permutations((2.0, 0.0, 0.0, 0.0)))

    # Group 2 (16 vertices): all sign combinations of (±1, ±1, ±1, ±1)
    vertices_list.extend(_signed_permutations((1.0, 1.0, 1.0, 1.0)))

    # Group 3 (96 vertices): even permutations of (±φ, ±1, ±1/φ, 0)
    vertices_list.extend(
        _signed_permutations((phi, 1.0, 1.0 / phi, 0.0), even_only=True)
    )

    vertices = np.array(vertices_list, dtype=np.float64)

    # Edge length = √5 − 1, edge_length² = 6 − 2√5
    edge_length_sq = 6.0 - 2.0 * np.sqrt(5)
    edges = _edges_by_distance(vertices, edge_length_sq)

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="600-cell")


def create_120cell() -> Polytope:
    """Create a 120-cell (hecatonicosachoron) centered at the origin.

    The 120-cell is the 4D analogue of the dodecahedron with 120
    dodecahedral cells.

    Returns:
        Polytope with 600 vertices and 1200 edges.
    """
    phi = (1 + np.sqrt(5)) / 2
    sqrt5 = np.sqrt(5)

    vertices_list: list[list[float]] = []

    # Set 1 (24 vertices): permutations of (±2, ±2, 0, 0)
    vertices_list.extend(_signed_permutations((2.0, 2.0, 0.0, 0.0)))

    # Set 2 (64 vertices): permutations of (±√5, ±1, ±1, ±1)
    vertices_list.extend(_signed_permutations((sqrt5, 1.0, 1.0, 1.0)))

    # Set 3 (64 vertices): permutations of (±φ⁻², ±φ, ±φ, ±φ)
    vertices_list.extend(
        _signed_permutations((1.0 / phi ** 2, phi, phi, phi))
    )

    # Set 4 (64 vertices): permutations of (±φ², ±φ⁻¹, ±φ⁻¹, ±φ⁻¹)
    vertices_list.extend(
        _signed_permutations((phi ** 2, 1.0 / phi, 1.0 / phi, 1.0 / phi))
    )

    # Set 5 (96 vertices): even permutations of (±φ², ±φ⁻², ±1, 0)
    vertices_list.extend(
        _signed_permutations(
            (phi ** 2, 1.0 / phi ** 2, 1.0, 0.0), even_only=True
        )
    )

    # Set 6 (96 vertices): even permutations of (±√5, ±φ⁻¹, ±φ, 0)
    vertices_list.extend(
        _signed_permutations((sqrt5, 1.0 / phi, phi, 0.0), even_only=True)
    )

    # Set 7 (192 vertices): even permutations of (±2, ±1, ±φ, ±φ⁻¹)
    vertices_list.extend(
        _signed_permutations((2.0, 1.0, phi, 1.0 / phi), even_only=True)
    )

    vertices = np.array(vertices_list, dtype=np.float64)

    # Edge length = 3 − √5, edge_length² = 14 − 6√5
    edge_length_sq = 14.0 - 6.0 * sqrt5
    edges = _edges_by_distance(vertices, edge_length_sq)

    # Normalise vertices to unit distance from origin
    vertices /= np.linalg.norm(vertices[0])

    return Polytope(vertices=vertices, edges=edges, name="120-cell")
