"""Rotation matrices and projection functions for 3D transformations."""

import numpy as np
from numpy.typing import NDArray


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
