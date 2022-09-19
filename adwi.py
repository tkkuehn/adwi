"""Read and assemble aDWI files."""

import io

import pandas as pd


def read_waveforms(path):
    """Read waveform file into a dataframe."""
    return pd.read_csv(
        path, sep=" ", skiprows=1, header=None, comment="#", names=["x", "y", "z"]
    )


def read_vols(path):
    """Read volume file into a dataframe."""
    return pd.read_csv(path, sep=":", comment="#", names=["waveform_idx"])


def read_dirs(path):
    """Parse and read direction file into a dataframe."""
    with open(path, "r", encoding="utf-8") as dir_file:
        lines = list(dir_file.readlines())
    return pd.read_csv(
        io.StringIO(
            "\n".join(
                [line.split(" = ")[1].strip()[1:-1].strip() for line in lines[3:]]
            )
        ),
        header=None,
        names=["x", "y", "z"],
    )
