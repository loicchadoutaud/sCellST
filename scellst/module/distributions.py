"""Code imported from scvi-tools.(https://github.com/YosefLab/scvi-tools/)"""

import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)


def log_zinb_positive(
    x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8
):
    """Log likelihood (scalar) of a minibatch according to a zinb model.

    Parameters
    ----------
    x
        Data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi
        logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps
        numerical stability constant

    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res


def log_nb_positive(
    x: Union[torch.Tensor],
    mu: Union[torch.Tensor],
    theta: Union[torch.Tensor],
    eps: float = 1e-6,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta + eps)
        - lgamma(theta + eps)
        - lgamma(x + 1)
    )
    return res


def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    r"""NB parameterizations conversion.

    Parameters
    ----------
    mu
        mean of the NB distribution.
    theta
        inverse overdispersion.
    eps
        constant used for numerical log stability. (Default value = 1e-6)

    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    if not (mu is None) == (theta is None):
        raise ValueError(
            "If using the mu/theta NB parameterization, both parameters must be specified"
        )
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def _convert_counts_logits_to_mean_disp(total_count, logits):
    """NB parameterizations conversion.

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    logits
        success logits.

    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.

    """
    theta = total_count
    mu = logits.exp() * total_count
    return mu, theta


def _gamma(theta, mu):
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = Gamma(concentration=concentration, rate=rate)
    return gamma_d


class NegativeBinomial(Distribution):
    r"""Negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "scale": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
        else:
            mu, theta = broadcast_all(mu, theta)
            total_count, logits = _convert_mean_disp_to_counts_logits(
                mu, theta, self._eps
            )
        self.mu = mu
        self.theta = theta
        self.scale = scale
        super().__init__(validate_args=validate_args)
        self.total_count = total_count
        self.logits = logits

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean**2) / self.theta

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: Optional[Union[torch.Size, Tuple]] = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = PoissonTorch(
            l_train
        ).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
        return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )

        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)

    def _gamma(self):
        return _gamma(self.theta, self.mu)


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    r"""Zero-inflated negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    zi_logits
        Logits scale of zero inflation probability.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "zi_logits": constraints.real,
        "scale": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        zi_logits: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        super().__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            mu=mu,
            theta=theta,
            scale=scale,
            validate_args=validate_args,
        )
        self.zi_logits, self.mu, self.theta = broadcast_all(
            zi_logits, self.mu, self.theta
        )

    @property
    def mean(self):
        pi = self.zi_probs
        return (1 - pi) * self.mu

    @property
    def variance(self):
        raise NotImplementedError

    @lazy_property
    def zi_logits(self) -> torch.Tensor:
        """ZI logits."""
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self) -> torch.Tensor:
        return logits_to_probs(self.zi_logits, is_binary=True)

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: Optional[Union[torch.Size, Tuple]] = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        samp = super().sample(sample_shape=sample_shape)
        is_zero = torch.rand_like(samp) <= self.zi_probs
        samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
        return samp_

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
            )
        return log_zinb_positive(value, self.mu, self.theta, self.zi_logits, eps=1e-08)
