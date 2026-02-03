"""Rotation matrices and projection functions for 3D and 4D transformations."""

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# 3D Rotation Matrices
# =============================================================================


def rotation_x(angle: float) -> NDArray[np.float64]:
    """Create a rotation matrix around the X-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_y(angle: float) -> NDArray[np.float64]:
    """Create a rotation matrix around the Y-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ],
        dtype=np.float64,
    )


def rotation_z(angle: float) -> NDArray[np.float64]:
    """Create a rotation matrix around the Z-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def apply_rotation(
    vertices: NDArray[np.float64], rx: float, ry: float, rz: float
) -> NDArray[np.float64]:
    """Apply combined rotation around X, Y, and Z axes.

    Rotation order: X → Y → Z (standard Euler angles).

    Args:
        vertices: Array of shape (N, 3) with 3D coordinates.
        rx: Rotation angle around X-axis in radians.
        ry: Rotation angle around Y-axis in radians.
        rz: Rotation angle around Z-axis in radians.

    Returns:
        Rotated vertices with the same shape as input.
    """
    # Combined rotation matrix: R = Rz @ Ry @ Rx
    rotation_matrix = rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)
    return vertices @ rotation_matrix.T


def orthogonal_project(vertices_3d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project 3D vertices to 2D using orthogonal projection.

    Simply drops the Z coordinate (projects onto XY plane).

    Args:
        vertices_3d: Array of shape (N, 3) with 3D coordinates.

    Returns:
        Array of shape (N, 2) with 2D coordinates.
    """
    return vertices_3d[:, :2]


# =============================================================================
# 4D Rotation Matrices (6 rotation planes)
# =============================================================================


def rotation_xy_4d(angle: float) -> NDArray[np.float64]:
    """Create a 4D rotation matrix in the XY plane.

    Rotates X and Y coordinates; W and Z remain fixed.

    Args:
        angle: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def rotation_xz_4d(angle: float) -> NDArray[np.float64]:
    """Create a 4D rotation matrix in the XZ plane.

    Rotates X and Z coordinates; W and Y remain fixed.

    Args:
        angle: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0, -s, 0],
            [0, 1, 0, 0],
            [s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def rotation_xw_4d(angle: float) -> NDArray[np.float64]:
    """Create a 4D rotation matrix in the XW plane.

    Rotates X and W coordinates; Y and Z remain fixed.

    Args:
        angle: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0, 0, -s],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [s, 0, 0, c],
        ],
        dtype=np.float64,
    )


def rotation_yz_4d(angle: float) -> NDArray[np.float64]:
    """Create a 4D rotation matrix in the YZ plane.

    Rotates Y and Z coordinates; W and X remain fixed.

    Args:
        angle: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def rotation_yw_4d(angle: float) -> NDArray[np.float64]:
    """Create a 4D rotation matrix in the YW plane.

    Rotates Y and W coordinates; X and Z remain fixed.

    Args:
        angle: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, 0, -s],
            [0, 0, 1, 0],
            [0, s, 0, c],
        ],
        dtype=np.float64,
    )


def rotation_zw_4d(angle: float) -> NDArray[np.float64]:
    """Create a 4D rotation matrix in the ZW plane.

    Rotates Z and W coordinates; X and Y remain fixed.

    Args:
        angle: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c],
        ],
        dtype=np.float64,
    )


def apply_rotation_4d(
    vertices: NDArray[np.float64],
    rxy: float,
    rxz: float,
    rxw: float,
    ryz: float,
    ryw: float,
    rzw: float,
) -> NDArray[np.float64]:
    """Apply combined rotation in all 6 planes of 4D space.

    Args:
        vertices: Array of shape (N, 4) with 4D coordinates.
        rxy: Rotation angle in XY plane in radians.
        rxz: Rotation angle in XZ plane in radians.
        rxw: Rotation angle in XW plane in radians.
        ryz: Rotation angle in YZ plane in radians.
        ryw: Rotation angle in YW plane in radians.
        rzw: Rotation angle in ZW plane in radians.

    Returns:
        Rotated vertices with the same shape as input.
    """
    # Order: ZW first so W-plane rotations can propagate through XZ/YZ to X/Y
    rotation_matrix = (
        rotation_xy_4d(rxy)
        @ rotation_xz_4d(rxz)
        @ rotation_yz_4d(ryz)
        @ rotation_xw_4d(rxw)
        @ rotation_yw_4d(ryw)
        @ rotation_zw_4d(rzw)
    )
    return vertices @ rotation_matrix.T


def orthogonal_project_4d(vertices_4d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project 4D vertices to 3D using orthogonal projection.

    Simply drops the W coordinate (projects onto XYZ hyperplane).

    Args:
        vertices_4d: Array of shape (N, 4) with 4D coordinates.

    Returns:
        Array of shape (N, 3) with 3D coordinates.
    """
    return vertices_4d[:, :3]


def perspective_project(
    vertices_3d: NDArray[np.float64],
    distance: float,
) -> NDArray[np.float64]:
    """Project 3D vertices to 2D with perspective projection.

    Camera is positioned at (0, 0, distance) looking toward the origin.

    Args:
        vertices_3d: Array of shape (N, 3) with 3D coordinates.
        distance: Distance from camera to origin along Z-axis.

    Returns:
        Array of shape (N, 2) with 2D coordinates.
    """
    z = vertices_3d[:, 2]
    # Clip vertices at/behind camera to avoid division issues
    z_safe = np.minimum(z, distance - 0.01)
    scale = distance / (distance - z_safe)
    return np.column_stack([
        vertices_3d[:, 0] * scale,
        vertices_3d[:, 1] * scale,
    ])


def perspective_project_4d(
    vertices_4d: NDArray[np.float64],
    distance: float,
) -> NDArray[np.float64]:
    """Project 4D vertices to 3D with perspective projection.

    Camera is positioned at (0, 0, 0, distance) looking toward the origin.

    Args:
        vertices_4d: Array of shape (N, 4) with 4D coordinates.
        distance: Distance from camera to origin along W-axis.

    Returns:
        Array of shape (N, 3) with 3D coordinates.
    """
    w = vertices_4d[:, 3]
    # Clip vertices at/behind camera to avoid division issues
    w_safe = np.minimum(w, distance - 0.01)
    scale = distance / (distance - w_safe)
    return np.column_stack([
        vertices_4d[:, 0] * scale,
        vertices_4d[:, 1] * scale,
        vertices_4d[:, 2] * scale,
    ])
