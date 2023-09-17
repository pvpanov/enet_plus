# -*- coding: utf-8 -*-

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Callable, Iterable, Optional, Union

# Third Party
import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds

__author__ = "Petr Panov"
__copyright__ = "Copyleft 2023, Milky Way"
__credits__ = ["Petr Panov"]
__license__ = "GNU"
__version__ = "0.1.0"
__maintainer__ = "Petr Panov"
__email__ = "pvpanov93@gmail.com"
__status__ = "Draft"


_logger = logging.getLogger("ENetPlusSettings")


@dataclass
class ElasticNetPlusSettings:
    """Configuration for the ElasticNetPlus linear model."""

    loss_fun: Union[
        KnownSignedLossFunctions,
        Callable[
            [npt.NDArray[np.float_], npt.NDArray[np.float_]], npt.NDArray[np.float_]
        ],
    ]
    a0: Optional[npt.NDArray[np.float_]]
    fit_intercept: bool
    eps: float
    l1_penalties: Optional[Union[float, npt.NDArray[np.float_]]]
    l1_indexes: Optional[Iterable[int]]
    l1_means: Optional[Union[float, npt.NDArray[np.float_]]]
    l2_penalties: Optional[Union[float, npt.NDArray[np.float_]]]
    l2_indexes: Optional[Iterable[int]]
    l2_means: Optional[Union[float, npt.NDArray[np.float_]]]
    upper_tail: float
    lower_tail: float
    demean_with_weighting: bool
    score_with_weighting: bool
    custom_penalty: Optional[Callable[[npt.NDArray[np.float_]], float]]
    maxiter: int
    bounds: Optional[
        Union[
            tuple[int, int],
            Iterable[tuple[int, int]],
            dict[int, tuple[int, int]],
            Bounds,
            Iterable[Bounds],
            dict[int, Bounds],
        ]
    ]
    tol: float
    lbfgsb_params: Optional[dict]

    def __post_init__(self):
        assert self.eps >= 0
        assert 0 <= self.lower_tail < 0.5 < 1 - self.upper_tail <= 1.0
        assert self.tol > 0
        self.base_loss = (
            _get_loss_fun(self.loss_fun)
            if isinstance(self.loss_fun, (str, KnownSignedLossFunctions))
            else self.loss_fun
        )

        self._with_left_huber = not np.isclose(self.lower_tail, 0)
        self._with_right_huber = not np.isclose(self.upper_tail, 0)
        self.with_huber = self._with_right_huber or self._with_left_huber
        self._demean_with_weighting = False
        if self.demean_with_weighting:
            if not self.fit_intercept:
                _logger.warning(
                    f"demean_with_weighting does nothing if fit_intercept=False"
                )
            elif not self.with_huber:
                _logger.warning(
                    f"demean_with_weighting does nothing if both tail params are 0"
                )
            else:
                self._demean_with_weighting = True
        if self.score_with_weighting and not self.with_huber:
            _logger.warning(
                "score_with_weighting does nothing if both tail params are 0"
            )
        self._init_fields = {
            f.name: getattr(self, f.name) for f in fields(ElasticNetPlusSettings)
        }

    def __repr__(self):
        return "ElasticNetPlusSettings\n" + "\n".join(
            ("{:>24}: {:<64}".format(k, str(v)) for k, v in self._init_fields.items())
        )

    def modify(self, **kwargs) -> ElasticNetPlusSettings:
        return ElasticNetPlusSettings(**{**self._init_fields, **kwargs})


def _quadratic_loss(
    target: npt.NDArray[np.float_], prediction: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    diff = target - prediction
    losses = diff * np.abs(diff)
    return losses


def _logistic_loss(
    *, target: npt.NDArray[np.float_], prediction: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    losses = -target * np.log(prediction)
    mask = np.isclose(target, 0.0)
    losses[mask] = np.log(1 - prediction[mask])
    return losses


class KnownSignedLossFunctions(enum.Enum):
    quadratic = "quadratic"
    logistic = "logistic"

    @classmethod
    def contains(cls, key):
        return key in cls.__members__


def _get_loss_fun(
    key: Union[str, KnownSignedLossFunctions]
) -> Callable[[npt.NDArray[np.float_], npt.NDArray[np.float_]], npt.NDArray[np.float_]]:
    if key == KnownSignedLossFunctions.quadratic:
        return _quadratic_loss  # type: ignore[return-value]
    elif key == KnownSignedLossFunctions.logistic:
        return _logistic_loss  # type: ignore[return-value]
    else:
        raise KeyError(f"unknown function type: {key}")
