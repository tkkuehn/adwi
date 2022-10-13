"""Read and assemble aDWI files."""

from __future__ import annotations
import io
from os import PathLike
import re

from attrs import define
import cbor2
import click
import numpy as np
import pandas as pd


def read_waveforms(path: PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read waveform file into a dataframe."""
    with open(path, "r", encoding="utf-8") as waveform_file:
        waveform_lines: list[str] = waveform_file.readlines()
    try:
        waveform_2_start: int | None = waveform_lines.index("XYDAT2= (T X Y Z)")
    except ValueError:
        waveform_2_start = None
    if waveform_2_start is None:
        waveform: pd.DataFrame = pd.read_csv(
            path, sep=" ", skiprows=1, header=None, comment="#", names=["x", "y", "z"]
        )
        return (waveform, waveform)
    waveform_1: pd.DataFrame = pd.read_csv(
        io.StringIO("\n".join(waveform_lines[:waveform_2_start])),
        skiprows=1,
        header=None,
        comment="#",
        names=["x", "y", "z"],
    )
    waveform_2: pd.DataFrame = pd.read_csv(
        io.StringIO("\n".join(waveform_lines[waveform_2_start + 1 :])),
        header=None,
        comment="#",
        names=["x", "y", "z"],
    )
    return (waveform_1, waveform_2)


def read_vols(path: PathLike) -> pd.DataFrame:
    """Read volume file into a dataframe."""
    return pd.read_csv(path, sep=":", comment="#", names=["vol_idx", "waveform_idx"])


def read_dirs(path: PathLike) -> pd.DataFrame:
    """Parse and read direction file into a dataframe."""
    dir_pattern = (
        r"Vector\[\d+\] = \( (-?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?) \)"
    )
    with open(path, "r", encoding="utf-8") as dir_file:
        lines: list[str] = list(dir_file.readlines())
    return pd.read_csv(
        io.StringIO(
            "\n".join([re.match(dir_pattern, line).group(1) for line in lines[3:]])
        ),
        header=None,
        names=["x", "y", "z"],
    )


def rot_matrix(u_vec, theta: float):
    """Calculate a rotation matrix from and axis of rotation and angle.

    Parameters
    ----------
    u_vec : array_like
        Unit vector representing the axis of rotation.
    theta : float
        Angle of rotation in radians.
    """
    return (
        np.cos(theta) * np.identity(3)
        + np.sin(theta) * np.cross(u_vec, np.identity(3) * -1)
        + (1 - np.cos(theta)) * np.outer(u_vec, u_vec)
    )


def angles_from_rot_matrix(matrix):
    """Calculate three Euler angles from a rotation matrix.

    Parameters
    ----------
    matrix : array_like
        3x3 rotation matrix.

    Returns
    -------
    array_like
        Extrinsic Euler angles about the x, y, and z axis in that order.
    """
    theta_y = np.arcsin(matrix[0, 2])
    theta_x = np.arctan2(-1 * matrix[1, 2], matrix[2, 2])
    theta_z = np.arctan2(-1 * matrix[0, 1], matrix[0, 0])

    return np.array([theta_x, theta_y, theta_z])


def angles_from_dir(orig, dir_vec):
    """Calculate Euler angles to rotate a vector to another.

    Parameters
    ----------
    orig : array_like
        Original vector to be rotated.
    dir_vec : array_like
        Target vector.

    Returns
    -------
    array_like
        X, Y, and Z rotations, in degrees.
    """
    cross = np.cross(orig, dir_vec)
    cross_norm = np.linalg.norm(cross)
    if cross_norm == 0:
        return np.array([0, 0, 0])
    u_vec = cross / cross_norm
    theta = np.arcsin(cross_norm / np.linalg.norm(dir_vec) / np.linalg.norm(orig))
    matrix = rot_matrix(u_vec, theta)
    return np.degrees(angles_from_rot_matrix(matrix))


def build_rot_table(orig, dirs):
    """Calculate Euler angles to rotate a vector to a series of vectors.

    Parameters
    ----------
    orig : array_like
        Original vector to be rotated.
    dirs : DataFrame
        n x 3 dataframe of target vectors.

    Returns
    -------
    DataFrame
        n x 3 dataframe of Euler angles.
    """

    def angles_from_row(row):
        return angles_from_dir(orig, row)

    return dirs.apply(angles_from_row, axis=1, result_type="broadcast")


@define
class BinaryWave:
    """Binary representation of a gradient waveform."""
    grad1: pd.DataFrame
    grad2: pd.DataFrame

    def as_dict(self) -> dict[str, list[float]]:
        """Represent the waveform as a dictionary."""
        return {
            "xgrad1": list(self.grad1.x),
            "ygrad1": list(self.grad1.y),
            "zgrad1": list(self.grad1.z),
            "xgrad2": list(self.grad2.x),
            "ygrad2": list(self.grad2.y),
            "zgrad2": list(self.grad2.z),
        }

    def write_cbor(self, out_path: PathLike):
        """Write the waveform dict to a CBOR file."""
        with open(out_path, "wb") as out_file:
            cbor2.dump(self.as_dict(), out_file)


class SpecificationError(Exception):
    """Error raised when the diffusion scheme is misspecified."""


@click.group()
def cli():
    """Generate aDWI-BIDS files from source inputs."""


@cli.command()
@click.argument("waveform_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def gen_waveform_cbor(waveform_path: PathLike, out_path: PathLike):
    """Read a waveform from a file and write a CBOR file describing it."""
    BinaryWave(*read_waveforms(waveform_path)).write_cbor(out_path)


@cli.command()
@click.argument("dirs_path", type=click.Path(exists=True))
@click.argument("vols_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def gen_dir_table(dirs_path: PathLike, vols_path: PathLike, out_path: PathLike):
    """Read directions from a file and write a file with rotations."""
    dirs = read_dirs(dirs_path)
    vols = read_vols(vols_path)
    if vols.shape[0] % dirs.shape[0] != 0:
        raise SpecificationError("Volume and direction table lengths don't match up.")
    dir_repetitions = vols.shape[0] // dirs.shape[0]
    orig = np.array([1, 0, 0])
    rot_table = build_rot_table(orig, dirs)
    out_table = pd.DataFrame(
        rot_table.values.repeat(dir_repetitions, axis=0), columns=rot_table.columns
    )
    out_table.insert(0, "v", vols["vol_idx"])
    out_table.insert(1, "d", vols["waveform_idx"])
    out_table.to_csv(out_path, sep="\t", index=False)
