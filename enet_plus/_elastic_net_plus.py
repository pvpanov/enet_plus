# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Optional, Union

# Third Party
import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, OptimizeResult
from sklearn.base import BaseEstimator, MultiOutputMixin

# ENetPlus
from enet_plus._settings import ElasticNetPlusSettings, KnownSignedLossFunctions

__author__ = "Petr Panov"
__copyright__ = "Copyleft 2023, Milky Way"
__credits__ = ["Petr Panov"]
__license__ = "GNU"
__version__ = "0.1.0"
__maintainer__ = "Petr Panov"
__email__ = "pvpanov93@gmail.com"
__status__ = "Draft"

_logger = logging.getLogger("ENetPlus")


def _check_data(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    sample_weight: Optional[npt.NDArray[np.float_]] = None,
):
    assert x.ndim == 2
    nx = len(x)
    assert np.all(~np.isinf(x))
    assert np.all(~np.isnan(x))

    assert y.ndim == 2
    assert len(y) == nx
    assert np.all(~np.isinf(y))
    assert np.all(~np.isnan(y))

    if sample_weight is not None:
        assert np.all(~np.isinf(sample_weight))
        assert np.all(~np.isnan(sample_weight))
        assert np.isclose(np.sum(sample_weight), 1.0)
        assert len(sample_weight) == nx


class ElasticNetPlus(MultiOutputMixin, BaseEstimator):
    """Linear regression/classification with non-central `L1` and `L2` priors as regularizer, bounds on learned coefficients,
    custom regression penalty, and Huber-type loss regularization. Uses L-BFGS to inimize the following objective function::

        .5 * WeightedHuberLoss(loss_fun; y - Xa; upper_tail, lower_tail)
        + sum_{i from i1} l_{i, 1} |a_i - v_{i, 1}|
        + .5 * sum_{i from i2} l_{i ,2} |a_i - v_{i, 2}|^2
        + custom_penalty(a)

    s.t. `a_i` satisfies bounds `b_i` for certain indexes `i`, and `sum(sample_weights) = 1`.

    Parameters
    ----------
    loss_fun
        `Signed` loss function that we optimize for (known pre-specified losses: "quadratic" and "logistic").
        Sign indicates whether we're above or below the target (used if we downweigh extreme observations asymmetrically).

    a0
        Warm start (starting point of the optimization). If boudns are provided, it must satisfy them. Does not include intercept.

    fit_intercept
        Whether to fit intercept (if True, it shall receive no penalty).

    eps
        Coefficients are hard-thresholded on this (setting it to 0 and using the L1 penalty can slow the optimization down).
        Also, an `option` for the `L-BFGS-B` optimizer.

    upper_tail
        Once the residual is above `1 - upper_tail` quantile of residuals, the samples are downweighted by the square
        root of the loss function, which is similar to how Huber loss downweighs big residuals. The parameter is ignored
        if somehow the upper tail is negative.

    lower_tail
        The loss on residuals becomes linear once the residual is below `lower_tail` quantile of residuals. The parameter is
        ignored if somehow the lower tail of residuals is positive.

    demean_with_weighting
        If fit_intercept=True and upper_tail or lower_tail are nonzero, this flags whether we weigh to compute the intercept.

    score_with_weighting : bool, default=False
        Whether to downweigh tail residuals when calculating the score.

    l1_penalties
        Controls the strength of the `L1` loss component. `None` means that the penalty will not be applied. If a `float`
        is provided, the penalty is assumed to be the same for all the coefficients. Parameter `eps` is used to smoothify
        this penalty near 0 (to avoid the Hessian blowing up).

    l1_indexes
        Controls which coefficients to apply the `L1` penalty to. `None` means it will be inferred from `l1_penalties`.

    l1_means
        Controls which coefficients towards what to shrink the coefficients with the `L1` regularization.
        `None` means it will be inferred from `l1_penalties`.

    l2_penalties
        Controls the strength of the `L2` loss component. `None` means that the penalty will not be applied. If a `float`
        is provided, the penalty is assumed to be the same for all the coefficients.

    l2_indexes
        Controls which coefficients to apply the `L2` penalty to. `None` means it will be inferred from `l2_penalties`.

    l2_means
        Controls which coefficients towards what to shrink the coefficients with the `L2` regularization.
        `None` means it will be inferred from `l2_penalties`.

    custom_penalty
        Optional extra penalty; if not `None`, it's still worth making it twice continuously differentiable near solution.

    maxiter
        Max number of iterations for given Huber weight modifications. One of the options for the optimizer.

    bounds
        Dictionary containing indexes of coefficients for which to apply bounds, and those bounds, with the latter being
        tuples or `scipy.optimize.Bounds`. Tuples are converted to `Bounds` with `keep_feasible=True`.
        If an iterable of tuples if provided, the ordering should match the coefficients'.
        If a single tuple is provided, the same bound is set for every coefficient.
        `None` is interpreted as no bounds.

    tol
        Tolerance for termination (the default here is 1e-6 whereas the default for L-BFGS-B is 1e-4).

    lbfgsb_params
        Special arguments that get passed in https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
        Note that `fun`, `x0`, `args`, `method`, `tol`, `bounds' are known given other params and shouldn't be provided.
        Similarly, `eps`, `maxiter` and `maxfun` for `options` are already provided.
        If you provide `jac` and hess` or `hessp

    Notes
    -----

    - The (pre-Huber) loss function can be anything monotonic on both semi-axes, but if `quadratic` is passed, the optimization can be faster.

    - The intercept does not get a special treatment.

    - For `quadratic` loss the Gram matrix is always precomputed.

    - Targets `y` are expected to be multivariate even though they get unstacked for optimization.

    - If `sample_weights` are not provided during the fit, they are set to be uniform, otherwise they are asserted to sum to `1`.

    - The coefficients in front of losses and penalties are s.t. we can easily match the behavior of
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    - Pass Fortran-continuous numpy arrays to the `fit` (`.values` from a typical `pandas.DataFrame` will do).

    - This algorithm optimizes the coefficients jointly.

    - Pass nonmutable arguments when possible.

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNet
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> regr = ElasticNet(random_state=0)
    >>> regr.fit(X, y)
    ElasticNet(random_state=0)
    >>> print(regr.coef_)
    [18.83816048 64.55968825]
    >>> print(regr.intercept_)
    1.451...
    >>> print(regr.predict([[0, 0]]))
    [1.451...]
    """

    def __init__(
        self,
        *,
        loss_fun: Union[
            KnownSignedLossFunctions,
            Callable[
                [npt.NDArray[np.float_], npt.NDArray[np.float_]], npt.NDArray[np.float_]
            ],
        ] = KnownSignedLossFunctions.quadratic,
        a0: Optional[npt.NDArray[np.float_]] = None,
        fit_intercept: bool = True,
        eps: float = 1e-6,
        l1_penalties: Optional[Union[float, npt.NDArray[np.float_]]] = None,
        l1_indexes: Optional[Iterable[int]] = None,
        l1_means: Optional[Union[float, npt.NDArray[np.float_]]] = None,
        l2_penalties: Optional[Union[float, npt.NDArray[np.float_]]] = None,
        l2_indexes: Optional[Iterable[int]] = None,
        l2_means: Optional[Union[float, npt.NDArray[np.float_]]] = None,
        upper_tail: float = 0.0,
        lower_tail: float = 0.0,
        demean_with_weighting: bool = True,
        score_with_weighting: bool = False,
        custom_penalty: Optional[Callable[[npt.NDArray[np.float_]], float]] = None,
        maxiter: int = 15000,
        bounds: Optional[
            Union[
                tuple[int, int],
                Iterable[tuple[int, int]],
                dict[int, tuple[int, int]],
                Bounds,
                Iterable[Bounds],
                dict[int, Bounds],
            ]
        ] = None,
        tol: float = 1e-6,
        lbfgsb_params: Optional[dict] = None,
    ):
        self._settings = ElasticNetPlusSettings(
            loss_fun=loss_fun,
            a0=a0,
            fit_intercept=fit_intercept,
            eps=eps,
            l1_penalties=l1_penalties,
            l1_indexes=l1_indexes,
            l1_means=l1_means,
            l2_penalties=l2_penalties,
            l2_indexes=l2_indexes,
            l2_means=l2_means,
            upper_tail=upper_tail,
            lower_tail=lower_tail,
            demean_with_weighting=demean_with_weighting,
            score_with_weighting=score_with_weighting,
            custom_penalty=custom_penalty,
            maxiter=maxiter,
            bounds=bounds,
            tol=tol,
            lbfgsb_params=lbfgsb_params,
        )
        self.coef_: Optional[npt.NDArray[np.float_]] = None
        self.intercept_: Union[npt.NDArray[np.float_], float] = 0.0
        self._path: Optional[list[OptimizeResult]] = None
        self._path_scores: Optional[list[float]] = None

    @property
    def path(self):
        return self._path

    @property
    def path_scores(self):
        return self._path_scores

    @property
    def settings(self) -> ElasticNetPlusSettings:
        return self._settings

    def fit(
        self,
        x: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_],
        sample_weight: Optional[npt.NDArray[np.float_]] = None,
    ) -> None:
        _check_data(x, y, sample_weight)
        self._path = []
        self._path_scores = []

        # append intercept if need be
        n, m = x.shape
        if self.settings.fit_intercept:
            x = np.column_stack([x, np.ones(n, dtype=np.float_)])
            m += 1
        # pick starting point
        if self.settings.a0 is None:
            self.coef_ = self._pick_a0(x, y, sample_weight)
        else:
            self.coef_ = self.settings.a0

    def _pick_a0(self, x, y, sample_weight):
        pass

    def _optimization_step(self, x, y, sample_weight):
        pass

    def predict(self, x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        assert self.coef_ is not None, "gotta fit this first, bud"
        return x @ self.coef_ + self.intercept_

    def _downweigh_big_residuals_inplace(self, resids: npt.NDArray[np.float_]):
        if not self.settings.with_huber:
            return
        ql, qr = np.quantile(
            resids,
            (self.settings.lower_tail, 1 - self.settings.upper_tail),
            method="weibull",
        )
        if ql < 0:
            mask = resids < ql
            resids[mask] = np.sqrt(resids[mask] * ql)
        if qr > 0:
            mask = resids > qr
            resids[mask] = np.sqrt(resids[mask] * qr)

    def score(
        self,
        x: npt.NDArray[np.float_],
        y: npt.NDArray[np.float_],
        sample_weight: Optional[npt.NDArray[np.float_]] = None,
    ):
        _check_data(x, y, sample_weight)
        signed_loss = self.settings.base_loss(y, self.predict(x))
        if self.settings.score_with_weighting:
            self._downweigh_big_residuals_inplace(resids=signed_loss)
        if sample_weight is not None:
            signed_loss *= sample_weight[:, None]
        return np.sum(signed_loss)

    def set_params(self, **params) -> ElasticNetPlus:
        if params:
            self._settings = self._settings.modify(**params)
        self.coef_ = None
        self.intercept_ = 0.0
        return self
