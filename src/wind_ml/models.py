"""Простые, объяснимые и воспроизводимые модели мощности."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_COLUMNS, feature_matrix


def persistence_predict(last_power: pd.Series | np.ndarray) -> np.ndarray:
    """Ограничить технический baseline естественным диапазоном [0, 1]."""

    return np.clip(np.asarray(last_power, dtype=float), 0.0, 1.0)


@dataclass
class PowerCurve:
    """Медианная бинированная кривая «фактический ветер → мощность».

    Это диагностическая и объяснимая модель. Для честного прогноза на ней
    следует подавать доступный в issue-time прогноз ветра, а не будущий SCADA
    ветер.
    """

    bin_width: float
    min_samples_per_bin: int
    wind_centers: list[float]
    median_power: list[float]
    counts: list[int]

    @classmethod
    def fit(
        cls,
        measurements: pd.DataFrame,
        *,
        bin_width: float = 0.5,
        min_samples_per_bin: int = 30,
    ) -> PowerCurve:
        complete = measurements.loc[
            measurements["quality"].eq("complete"), ["wind_speed", "power"]
        ].dropna()
        if bin_width <= 0:
            raise ValueError("bin_width должен быть положительным")
        if len(complete) < min_samples_per_bin:
            raise ValueError("Недостаточно полных часов для power curve")
        bins = np.floor(complete["wind_speed"].to_numpy() / bin_width).astype(int)
        grouped = (
            pd.DataFrame({"bin": bins, "power": complete["power"].to_numpy()})
            .groupby("bin")["power"]
            .agg(["median", "size"])
        )
        grouped = grouped.loc[grouped["size"] >= min_samples_per_bin]
        if len(grouped) < 2:
            raise ValueError("Нужно не менее двух заполненных бинов power curve")
        return cls(
            bin_width=bin_width,
            min_samples_per_bin=min_samples_per_bin,
            wind_centers=((grouped.index.to_numpy() + 0.5) * bin_width).astype(float).tolist(),
            median_power=grouped["median"].clip(0, 1).astype(float).tolist(),
            counts=grouped["size"].astype(int).tolist(),
        )

    def predict(self, wind_speed: pd.Series | np.ndarray) -> np.ndarray:
        wind = np.asarray(wind_speed, dtype=float)
        prediction = np.interp(
            wind,
            np.asarray(self.wind_centers),
            np.asarray(self.median_power),
            left=self.median_power[0],
            right=self.median_power[-1],
        )
        return np.clip(prediction, 0.0, 1.0)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PowerCurve:
        return cls(**payload)  # type: ignore[arg-type]


@dataclass
class BoostingModel:
    """Обёртка, фиксирующая порядок признаков вместе со sklearn-моделью."""

    estimator: HistGradientBoostingRegressor
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def predict(self, examples: pd.DataFrame) -> np.ndarray:
        return np.clip(self.estimator.predict(feature_matrix(examples)), 0.0, 1.0)


def fit_boosting(examples: pd.DataFrame, *, random_state: int = 42) -> BoostingModel:
    features = feature_matrix(examples)
    target = examples["target_power"].to_numpy(dtype=float)
    if len(features) < 100:
        raise ValueError("Для бустинга требуется не менее 100 честных обучающих строк")
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_leaf_nodes=24,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=random_state,
        early_stopping=False,
    )
    model.fit(features, target)
    return BoostingModel(model)
