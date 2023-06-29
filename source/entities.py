# from dataclasses import dataclass
# import shelve
import os
import re
from pathlib import PurePath
from typing import Any, Protocol, Callable, TypeVar

import pandas as pd
import lasio

# import dlisio
import matplotlib.pyplot as plt


PATH = PurePath(__file__).parent.parent
LOG_INFO = {
    "CALI": {
        "name": "Caliper",
        "min": 6,
        "max": 16,
        "aliases": {
            "LFP_CALI",
        },
    },
    "GR": {
        "name": "Gamma Ray",
        "min": 0,
        "max": 200,
        "aliases": {
            "LFP_GR",
        },
    },
    "RT": {
        "name": "Resistivity",
        "min": 0,
        "max": 0,
        "aliases": {
            "LFP_RT",
        },
    },
    "DEN": {
        "name": "Density",
        "min": 0,
        "max": 0,
        "aliases": {
            "LFP_RHOB_LOG",
        },
    },
    "NUE": {
        "name": "Neutron",
        "min": 0,
        "max": 0,
        "aliases": {
            "LFP_NPHI",
        },
    },
    "DT": {
        "name": "Sonic",
        "min": 0,
        "max": 0,
        "aliases": {
            "LFP_DT_LOG",
        },
    },
    "DEPTH": {
        "name": "Depth",
        "min": 0,
        "max": 0,
        "aliases": {
            "DEPTH",
        },
    },
}


class DataPaths:
    def __init__(self) -> None:
        self.las_file_paths: tuple[str, ...] = (
            str(PATH.joinpath(r"data\spwla_volve\15_9-19A\15_9-19_A_CPI.las")),
            str(PATH.joinpath(r"data\spwla_volve\15_9-19A\159-19A_LFP.las")),
        )
        self.lis_file_paths: tuple[str, ...] = ()
        self.dlis_file_paths: tuple[str, ...] = ()
        self.excel_file_paths: tuple[str, ...] = ()

    @staticmethod
    def extract_filename(path: str) -> str:
        filename = os.path.basename(path)
        if "." in filename:
            p = r"(.+)\.\w+"
            match = re.search(p, filename)
            return filename if match is None else match.group(1)
        return filename


class Log:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        name: str,
        style: str | None = "seaborn-v0_8-paper",
    ) -> None:
        if style is not None:
            plt.style.use(style)
        dataframe = dataframe.reset_index()
        self.dataframe = dataframe
        self.name = name

    def __repr__(self) -> str:
        return f"< Log {self.name} >"

    def to_excel(self) -> None:
        path = PATH.joinpath(f"data\\excel_files\\{self.name}.xlsx")
        self.dataframe.to_excel(str(path), index=False)

    @property
    def df(self) -> pd.DataFrame:
        return self.dataframe

    @classmethod
    def from_excel(cls, path: str) -> "Log":
        filename = DataPaths.extract_filename(path)
        try:
            df = pd.read_excel(path)
        except Exception:
            raise ValueError(f"{path} is not a correct path")
        return cls(dataframe=df, name=filename)

    @classmethod
    def from_las(cls, path: str) -> "Log":
        filename = DataPaths.extract_filename(path)
        try:
            las_object = lasio.read(path, engine="normal")
        except Exception:
            raise ValueError(f"{path} is not a correct path")
        df = las_object.df()
        return cls(dataframe=df, name=filename)

    @classmethod
    def from_lis(cls, path: str) -> "Log":
        ...

    @classmethod
    def from_dlis(cls, path: str) -> "Log":
        ...

    @staticmethod
    def join(*logs: "Log") -> "Log":
        ...

    @staticmethod
    def handle_outliers(log: "Log") -> None:
        ...

    @staticmethod
    def handle_missing(log: "Log") -> None:
        ...

    @staticmethod
    def rename_columns(log: "Log") -> None:
        ...

    @staticmethod
    def trim_columns(log: "Log") -> None:
        ...

    @staticmethod
    def confirm_columns(log: "Log") -> None:
        if False:
            raise ValueError(f"{log} format is invalid")

    @staticmethod
    def normalize(log: "Log") -> None:
        ...

    def log_plot(self, show: bool = False, annotate: bool = False):
        Log.confirm_columns(self)

        df = self.dataframe
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
        top, bottom = df["DEPTH"].iloc[0], df["DEPTH"].iloc[-1]

        ax[0].plot("LFP_GR", "DEPTH", data=df, color="black", label="Gamma")
        ax[0].set_xlabel("Gamma")
        ax[0].set_xlim(0, 300)

        cali = ax[0].twiny()
        cali.plot("LFP_CALI", "DEPTH", data=df, color="purple", label="Caliper")
        cali.set_xlabel("Caliper", color="purple")
        cali.set_xlim(6, 16)
        cali.tick_params(axis="x", colors="purple")
        cali.spines["top"].set_position(("axes", 1.08))
        cali.spines["top"].set_color("purple")

        ax[1].plot(
            "LFP_RT", "DEPTH", data=df, color="black", label="medium resistivity"
        )
        ax[1].set_xlabel("resistivity")
        ax[1].set_xlim(0.1, 2000)
        ax[1].semilogx()

        ax[2].plot("LFP_RHOB_LOG", "DEPTH", data=df, color="black")
        ax[2].set_xlabel("Density")
        ax[2].set_xlim(1.95, 2.95)

        nue = ax[2].twiny()
        nue.plot("LFP_NPHI", "DEPTH", data=df, color="purple", label="neutron")
        nue.set_xlabel("Neutron", color="purple")
        nue.set_xlim(0.45, -0.15)
        nue.tick_params(axis="x", colors="purple")
        nue.spines["top"].set_position(("axes", 1.08))
        nue.spines["top"].set_color("purple")

        ax[3].plot("LFP_DT_LOG", "DEPTH", data=df, color="black")
        ax[3].set_xlabel("Sonic")
        ax[3].set_xlim(140, 40)

        if annotate:
            oil = ax[3].twiny()
            oil.step(df["LFP_OIL"], df["DEPTH"], where="post", color="none")
            oil.fill_betweenx(
                df["DEPTH"], df["LFP_OIL"], step="post", alpha=0.4, color="green"
            )
            oil.set_xticks([])

            water = ax[3].twiny()
            water.step(df["LFP_WATER"], df["DEPTH"], where="post", color="none")
            water.fill_betweenx(
                df["DEPTH"], df["LFP_WATER"], step="post", alpha=0.4, color="blue"
            )
            water.set_xticks([])

            gas = ax[3].twiny()
            gas.step(df["LFP_GAS"], df["DEPTH"], where="post", color="none")
            gas.fill_betweenx(
                df["DEPTH"], df["LFP_GAS"], step="post", alpha=0.4, color="red"
            )
            gas.set_xticks([])

        for index, axi in enumerate(ax):
            axi.set_ylim(bottom, top)
            axi.xaxis.set_ticks_position("top")
            axi.xaxis.set_label_position("top")
            if index != 0:
                plt.setp(axi.get_yticklabels(), visible=False)

        fig.subplots_adjust(wspace=0.15)

        if show:
            plt.show()

    def box_plot(self, show: bool = False):
        ...


PIPELINE_VAR = TypeVar("PIPELINE_VAR")


class Pipeline:
    def __init__(self, *ops: Callable[[PIPELINE_VAR], PIPELINE_VAR]) -> None:
        self.operations = ops

    def __call__(self, arg: PIPELINE_VAR) -> PIPELINE_VAR:
        for func in self.operations:
            arg = func(arg)
        return arg


class RandomForest:
    name: str = "RandomForest"
    paths: DataPaths = DataPaths()
    model: Any | None

    def train(self):
        ...

    def test(self):
        ...

    def validate(self):
        ...

    def save_model(self):
        ...

    def optimize_parameters(self):
        ...

    def build(self, save: bool = True):
        ...

    def predict(self, data: Log) -> Log:
        ...


class Model(Protocol):
    def predict(self, data: Log) -> Log:
        raise NotImplementedError


class Loggy:
    def __init__(self, **models: Model) -> None:
        self.models = models

    def get_models(self) -> list[Model]:
        ...

    def interprete(self, log: Log, model_name: str) -> Log:
        ...

    def evaluate_model():
        ...


if __name__ == "__main__":
    ...
