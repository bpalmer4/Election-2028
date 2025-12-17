"""Tools for doing Bayesian aggregation of polls"""

from typing import Any, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[import-untyped]
import pytensor.tensor as pt

from common import MIDDLE_DATE, ensure


# --- Data preparation
TAIL_CENTRED = "tail_centred"


def _check_he_constraints(box: dict[str, Any]) -> None:
    """Check that the house effect constraints are as expected.
    The list of firms should be in the correct order, with the
    pollsters excluded from the sum-to-zero constraint should
    be at the end of the list."""

    # -- get the data we need to check
    he_sum_exclusions = box["he_sum_exclusions"]
    he_sum_inclusions = box["he_sum_inclusions"]
    firms = box["firm_list"]

    # -- check that our house effects are all lists of strings
    for check in (he_sum_exclusions, he_sum_inclusions, firms):
        ensure(
            isinstance(check, list),
            "House effect constraints must be lists of strings.",
        )
        for element in check:
            ensure(
                isinstance(element, str),
                "House effect constraints must be lists of strings.",
            )

    # -- check we have at least minimum_houses of constraints
    minimum_houses = 2
    ensure(
        len(he_sum_inclusions) >= minimum_houses,
        f"Need at least {minimum_houses} firm for house effects.",
    )

    # -- check the includesions are first
    he_inclusions2 = firms[: len(he_sum_inclusions)]
    ensure(
        set(he_sum_inclusions) == set(he_inclusions2),
        "The house-effect-constrained pollsters should be first in the 'firm-list'.",
    )

    # -- check the exclusions are last
    he_sum_exclusions2 = firms[-len(he_sum_exclusions) :]
    ensure(
        set(he_sum_exclusions) == set(he_sum_exclusions2),
        "The unconstrained pollsters should be last in the 'firm-list'.",
    )


def prepare_data_for_analysis(
    df: pd.DataFrame,
    column: str,
    **kwargs,
) -> dict[str, Any]:
    """Prepare a dataframe column for Bayesian analysis.
    Returns a python dict with all the necessary values within."""

    ensure(column in df.columns, "Column not found in DataFrame.")
    box: dict[str, Any] = {}  # container we will return
    box["column"] = column
    df = df.copy().loc[df[column].notnull()]  # remove nulls

    # make sure data is properly sorted by date, with an unique index
    df = df.sort_values(MIDDLE_DATE).reset_index(drop=True)
    box["df"] = df

    # get our zero centered observations, ignore missing data
    # assume data in percentage points (0..100)
    y = df[column]
    box["y"] = y

    if n := kwargs.get(TAIL_CENTRED, 0):
        # centre around last n polls
        # to minimise mean-reversion issues with GP, but may introduce bias
        centre_offset = -y.iloc[-n:].mean()
    else:
        centre_offset = -y.mean()
    zero_centered_y = y + centre_offset
    box["centre_offset"] = centre_offset
    box["zero_centered_y"] = zero_centered_y
    box["n_polls"] = len(zero_centered_y)

    # get our day-to-date mapping
    right_anchor: Optional[tuple[pd.Period, float]] = kwargs.get("right_anchor", None)
    box["right_anchor"] = right_anchor
    left_anchor: Optional[tuple[pd.Period, float]] = kwargs.get("left_anchor", None)
    box["left_anchor"] = left_anchor
    day_zero = pd.Period(
        left_anchor[0] if left_anchor is not None else df[MIDDLE_DATE].min(),
        freq="D",
    )
    box["day_zero"] = day_zero
    last_day = pd.Period(
        right_anchor[0] if right_anchor is not None else df[MIDDLE_DATE].max(),
        freq="D",
    )
    box["last_day"] = last_day
    poll_date = pd.Series(
        [pd.Period(x, freq="D") for x in df[MIDDLE_DATE]], index=df.index
    )
    box["poll_date"] = poll_date
    poll_day = pd.Series([(x - day_zero).n for x in poll_date], index=df.index)
    box["poll_day"] = poll_day
    box["n_days"] = poll_day.max() + 1
    box["poll_day_c_"] = np.c_[poll_day]  # numpy column vector of poll_days for GP

    # sanity checks - anchors must be before or after polling data
    ensure(
        (left_anchor and day_zero < df[MIDDLE_DATE].min())
        or (right_anchor and last_day > df[MIDDLE_DATE].max())
        or (not left_anchor and not right_anchor),
        "Anchors must be outside of the range of polling dates.",
    )

    # get house effects inputs
    empty_list: list[str] = []
    he_sum_exclusions: list[str] = kwargs.get("he_sum_exclusions", empty_list)
    missing_firm: list[str] = [
        e for e in he_sum_exclusions if e not in df.Brand.unique()
    ]
    if missing_firm:
        # firm is not in the data, but it is one we should exclude?
        he_sum_exclusions = sorted(list(set(he_sum_exclusions) - set(missing_firm)))
    box["he_sum_exclusions"] = he_sum_exclusions
    he_sum_inclusions: list[str] = [
        e for e in df.Brand.unique() if e not in he_sum_exclusions
    ]
    box["he_sum_inclusions"] = he_sum_inclusions

    # get pollster map - ensure polsters at end of the list are the excluded ones
    firm_list = he_sum_inclusions + he_sum_exclusions  # ensure inclusions first
    box["firm_list"] = firm_list
    ensure(
        len(firm_list) == len(set(firm_list)),
        "Remove duplicate pollsters in the firm_list.",
    )
    n_firms = len(firm_list)
    box["n_firms"] = n_firms
    ensure(n_firms > len(he_sum_exclusions), "Number of exclusions == number of firms.")
    firm_map = {firm: code for code, firm in enumerate(firm_list)}
    box["firm_map"] = firm_map
    box["back_firm_map"] = {v: k for k, v in firm_map.items()}
    box["poll_firm_number"] = pd.Series([firm_map[b] for b in df.Brand], index=df.index)

    # final sanity checks ...
    _check_he_constraints(box)

    # Information
    if kwargs.get("verbose", False):
        print(box)

    return box


# --- Bayesian models and model components
# - Gaussian Random Walk
def _guess_start(inputs: dict[str, Any], n=10) -> np.float64:
    """Guess a starting point for the random walk,
    based on the first n poll results."""

    if n > (m := len(inputs["zero_centered_y"])):
        n = m
        print(f"Caution: Input data series is only {n} observations long.")
    return inputs["zero_centered_y"][:n].mean()


def temporal_model(
    inputs: dict[str, Any], model: pm.Model, **kwargs
) -> pt.TensorVariable:
    """The temporal (hidden daily voting intention) model component.
    Used in Gaussian Random Walk (GRW and GRWLA) models.
    Note: setting the innovation through a prior
    often results in a model that needs many samples to
    overcome poor traversal of the posterior."""

    # check for specified parameters
    innovation = kwargs.get("innovation", None)

    # construct the temporal model
    with model:
        if innovation is None:
            # this does not mix well, but I haven't found anything that does
            beta_hint = {"alpha": 3, "beta": 0.5}
            print(f"innovation InverseGamma prior: {beta_hint}")
            innovation = pm.InverseGamma("innovation", **beta_hint)

        init_guess_sigma = 5.0  # SD for initial guess
        start_dist = pm.Normal.dist(mu=_guess_start(inputs), sigma=init_guess_sigma)

        voting_intention = pm.GaussianRandomWalk(
            "voting_intention",
            mu=0,  # no drift in this model
            sigma=innovation,
            init_dist=start_dist,
            steps=inputs["n_days"],
        )
    return voting_intention


def house_effects_model(inputs: dict[str, Any], model: pm.Model) -> pt.TensorVariable:
    """The house effects model. This model component is used with
    both the GRW and the Gaussian Process (GP) models."""

    house_effect_sigma = inputs.get("house_effect_sigma", 5.0)
    with model:
        if inputs["right_anchor"] is None and inputs["left_anchor"] is None:
            if len(inputs["he_sum_exclusions"]) > 0:
                # sum to zero constraint for some (but not all) houses
                zero_sum_he = pm.ZeroSumNormal(
                    "zero_sum_he",
                    sigma=house_effect_sigma,
                    shape=len(inputs["he_sum_inclusions"]),
                )
                unconstrained_he = pm.Normal(
                    "unconstrained_he",
                    sigma=house_effect_sigma,
                    shape=len(inputs["he_sum_exclusions"]),
                )
                house_effects = pm.Deterministic(
                    "house_effects",
                    var=pm.math.concatenate([zero_sum_he, unconstrained_he]),
                )
            else:
                # sum to zero constraint for all houses
                house_effects = pm.ZeroSumNormal(
                    "house_effects", sigma=house_effect_sigma, shape=inputs["n_firms"]
                )
        else:
            # all house effects are unconstrained, used in anchored models
            house_effects = pm.Normal(
                "house_effects", sigma=house_effect_sigma, shape=inputs["n_firms"]
            )
    return house_effects


def core_likelihood(
    inputs: dict[str, Any],
    model: pm.Model,
    voting_intention: pt.TensorVariable,
    house_effects: pt.TensorVariable,
    **kwargs,
) -> None:
    """Likelihood for both GP and GRW models. But you
    must pass grw=False for GP analysis."""

    # check for specified parameters
    likelihood = kwargs.get("likelihood", "Normal")
    nu = kwargs.get("nu", None)
    sigma_likelihood = kwargs.get("sigma_likelihood", None)
    grw = kwargs.get("grw", True)

    # construct likelihood
    with model:
        if sigma_likelihood is None:
            sigma_likelihood_hint = {"sigma": 5}
            print(f"sigma_likelihood HalfNormal prior: {sigma_likelihood_hint}")
            sigma_likelihood = pm.HalfNormal(
                "sigma_likelihood", **sigma_likelihood_hint
            )
        mu = (
            voting_intention[inputs["poll_day"]]
            + house_effects[inputs["poll_firm_number"]]
            if grw
            else voting_intention + house_effects[inputs["poll_firm_number"]]
        )
        common_args = {
            "name": "observed_polls",
            "mu": mu,
            "sigma": sigma_likelihood,
            "observed": inputs["zero_centered_y"],
        }

        match likelihood:
            case "Normal":
                pm.Normal(**common_args)

            case "StudentT":
                if nu is None:
                    nu_hint = {"alpha": 2, "beta": 0.1}
                    print(f"nu Gamma prior: {nu_hint}")
                    nu = pm.Gamma("nu", **nu_hint) + 1

                pm.StudentT(
                    **common_args,
                    nu=nu,
                )


def grw_model(inputs: dict[str, Any], **kwargs) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls,
    using a Gaussian random walk (GRW). Model assumes poll data
    (in percentage points) has been zero-centered (by
    subtracting the mean for the series). Model assumes house
    effects sum to zero."""

    model = pm.Model()
    voting_intention = temporal_model(inputs, model, **kwargs)
    house_effects = house_effects_model(inputs, model)
    core_likelihood(inputs, model, voting_intention, house_effects, **kwargs)
    return model


# - Gaussian Process model ...
def gp_prior(
    inputs: dict[str, Any],
    model: pm.Model,
    **kwargs,
) -> pt.TensorVariable:
    """Construct the Gaussian Process (GP) latent variable model prior.
    The prior reflects voting intention on specific polling days.

    Note: Reasonably smooth looking plots only emerge with a lenjth_scale
    greater than (say) 15. Divergences occur when eta resolves as being
    close to zero, (which is obvious when you think about it, but also
    harder to avoid with series that are fairly flat). To address
    these sampling issues, we give the gamma distribution a higher alpha,
    as the mean of the gamma distribution is a/b. And we truncate eta to
    well avoid zero (noting eta is squared before being multiplied by the
    covariance matrix).

    Also note: for quick test runs, length_scale and eta can be fixed
    to (say) 40 and 1.6 respectively. With both specified, the model runs
    in around 1.4 seconds. With one or both unspecified, it takes about
    20 minutes per run, sometimes with divergences."""

    # check for specified parameters
    length_scale = kwargs.get("length_scale", None)
    eta = kwargs.get("eta", None)
    eta_prior = kwargs.get("eta_prior", "HalfCauchy")

    # construct the gaussian prior
    with model:
        if length_scale is None:
            # a length_scale around 40 appears to work reasonably
            gamma_hint = {"alpha": 40, "beta": 1}
            print(f"length_scale Gamma prior: {gamma_hint}")
            length_scale = pm.Gamma("length_scale", **gamma_hint)
            # Note: with the exponentiated quadratic kernel (without eta) ...
            #       at ls, correlation is around 0.61,
            #       at 2 * ls, correlation is around 0.14,
            #       at 3 * ls, correlation is around 0.01, etc.
            #       https://stats.stackexchange.com/questions/445484/

        if eta is None:
            # an eta around 1.6 appers to work well
            hint, function = {
                # I think HalfCauchy works best of those below
                "HalfCauchy": ({"beta": 4}, pm.HalfCauchy),
                "Gamma": ({"alpha": 2, "beta": 1}, pm.Gamma),
                "TruncatedNormal": (
                    {"lower": 0.5, "upper": 1000, "mu": 1.6, "sigma": 3},
                    pm.TruncatedNormal,
                ),
            }.get(eta_prior, ({"beta": 4}, pm.HalfNormal))
            print(f"eta {function.__name__} prior: {hint}")
            eta = function("eta", **hint)

        cov = (eta**2) * pm.gp.cov.ExpQuad(input_dim=1, ls=length_scale)
        gp = pm.gp.Latent(cov_func=cov)
        voting_intention = gp.prior("voting_intention", X=inputs["poll_day_c_"])
    return voting_intention


def gp_model(inputs: dict[str, Any], **kwargs) -> pm.Model:
    """PyMC model for pooling/aggregating voter opinion polls,
    using a Gaussian Process (GP). Note: **kwargs allows one to
    pass length_scale and eta to gp_prior() and/or pass approach,
    nu and sigma to gp_likelihood()."""

    model = pm.Model()
    voting_intention = gp_prior(inputs, model, **kwargs)
    house_effects = house_effects_model(inputs, model)
    core_likelihood(inputs, model, voting_intention, house_effects, **kwargs)
    return model


# --- Condition the model on the data
# Diagnostic thresholds
MAX_R_HAT = 1.01
MIN_ESS = 400
MAX_MCSE_RATIO = 0.05
MAX_DIVERGENCE_RATE = 1 / 10_000
MAX_TREE_DEPTH_RATE = 0.05
MIN_BFMI = 0.3


def check_model_diagnostics(
    trace: az.InferenceData,
    verbose: bool = True,
) -> list[str]:
    """Check the inference data for potential problems.

    Diagnostics applied:
    - R-hat (Gelman-Rubin): Compares between-chain and within-chain variance.
      Values > 1.01 suggest chains have not converged to the same distribution.
    - ESS (Effective Sample Size): Estimates independent samples accounting for
      autocorrelation. Low ESS (< 400) indicates high autocorrelation or short chains.
    - MCSE/sd ratio: Monte Carlo standard error relative to posterior sd.
      Ratios > 5% suggest insufficient samples for reliable posterior mean estimates.
    - Divergent transitions: Indicate regions where the sampler struggled with
      posterior geometry. Rate > 1/10,000 may signal biased estimates.
    - Tree depth saturation: High rates at max tree depth suggest the sampler
      is working harder than expected, possibly due to difficult geometry.
    - BFMI (Bayesian Fraction of Missing Information): Measures how well the
      sampler explores the energy distribution. Values < 0.3 suggest poor exploration.

    Args:
        trace: InferenceData from model fitting
        verbose: If True, print diagnostic results

    Returns:
        List of critical issues found (empty if none).
    """
    issues: list[str] = []
    summary = az.summary(trace)

    # Check model convergence (R-hat)
    statistic = summary.r_hat.max()
    is_problem = statistic > MAX_R_HAT
    if verbose:
        prefix = "WARNING: " if is_problem else ""
        print(f"{prefix}Maximum R-hat: {statistic:.4f}")
    if is_problem:
        issues.append(f"R-hat: {statistic:.4f}")

    # Check effective sample size
    statistic = summary[["ess_tail", "ess_bulk"]].min().min()
    is_problem = statistic < MIN_ESS
    if verbose:
        prefix = "WARNING: " if is_problem else ""
        print(f"{prefix}Minimum ESS: {int(statistic)}")
    if is_problem:
        issues.append(f"ESS: {int(statistic)}")

    # Check MCSE ratio (should be < 5% of posterior sd)
    statistic = (summary["mcse_mean"] / summary["sd"]).max()
    is_problem = statistic > MAX_MCSE_RATIO
    if verbose:
        prefix = "WARNING: " if is_problem else ""
        print(f"{prefix}Maximum MCSE/sd ratio: {statistic:.3f}")
    if is_problem:
        issues.append(f"MCSE/sd: {statistic:.3f}")

    # Check for divergences (rate-based: < 1 in 10,000 samples)
    try:
        diverging_count = int(np.sum(trace.sample_stats.diverging))
    except (ValueError, AttributeError):
        diverging_count = 0
    total_samples = trace.posterior.sizes["draw"] * trace.posterior.sizes["chain"]
    divergence_rate = diverging_count / total_samples
    is_problem = divergence_rate > MAX_DIVERGENCE_RATE
    if verbose:
        prefix = "WARNING: " if is_problem else ""
        print(
            f"{prefix}Divergent transitions: "
            f"{diverging_count}/{total_samples} ({divergence_rate:.4%})"
        )
    if is_problem:
        issues.append(f"Divergences: {diverging_count}")

    # Check max tree depth saturation
    try:
        if hasattr(trace.sample_stats, "reached_max_treedepth"):
            at_max = trace.sample_stats.reached_max_treedepth.values
            at_max_rate = float(at_max.mean())
            max_observed = int(trace.sample_stats.tree_depth.values.max())
            is_problem = at_max_rate >= MAX_TREE_DEPTH_RATE
            if verbose:
                prefix = "WARNING: " if is_problem else ""
                print(
                    f"{prefix}Tree depth at configured max: "
                    f"{at_max_rate:.2%} (max observed: {max_observed})"
                )
            if is_problem:
                issues.append(f"Tree depth: {at_max_rate:.2%}")
        else:
            ignore_threshold = 10
            tree_depth = trace.sample_stats.tree_depth.values
            max_depth = int(tree_depth.max())
            if max_depth >= ignore_threshold:
                at_max_rate = float((tree_depth == max_depth).mean())
                is_problem = at_max_rate >= MAX_TREE_DEPTH_RATE
                if verbose:
                    prefix = "WARNING: " if is_problem else ""
                    print(
                        f"{prefix}Tree depth at max ({max_depth}): "
                        f"{at_max_rate:.2%} (comparing to observed max)"
                    )
                if is_problem:
                    issues.append(f"Tree depth: {at_max_rate:.2%}")
            elif verbose:
                print("Tree depth check skipped (max depth too low).")
    except AttributeError:
        pass

    # Check BFMI
    statistic = az.bfmi(trace).min()
    is_problem = statistic < MIN_BFMI
    if verbose:
        prefix = "WARNING: " if is_problem else ""
        print(f"{prefix}Minimum BFMI: {statistic:.2f}")
    if is_problem:
        issues.append(f"BFMI: {statistic:.2f}")

    return issues


def draw_samples(model: pm.Model, **kwargs) -> tuple[az.InferenceData, str]:
    """Draw samples from the posterior distribution (ie. run the model)."""

    plot_trace = kwargs.pop("plot_trace", True)
    with model:
        idata = pm.sample(
            progressbar=True,
            return_inferencedata=True,
            **kwargs,
        )
        if plot_trace:
            az.plot_trace(idata)
    issues = check_model_diagnostics(idata, verbose=True)
    glitches = f"Sampling issues: {', '.join(issues)}" if issues else ""
    return (idata, glitches)
