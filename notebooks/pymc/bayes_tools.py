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

    # -- check the exclusions are last (skip if no exclusions)
    if len(he_sum_exclusions) > 0:
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


def _zero_median_transform(values: pt.TensorVariable) -> pt.TensorVariable:
    """Transform values so their median is zero.

    For odd n: median is the middle element after sorting.
    For even n: median is average of two middle elements.
    """
    n = values.shape[0]
    sorted_values = pt.sort(values)
    mid = n // 2
    median = pt.switch(
        pt.eq(n % 2, 0),
        (sorted_values[mid - 1] + sorted_values[mid]) / 2,
        sorted_values[mid],
    )
    return values - median


def house_effects_model(
    inputs: dict[str, Any], model: pm.Model, constraint: str = "zero_median"
) -> pt.TensorVariable:
    """The house effects model. This model component is used with
    both the GRW and the Gaussian Process (GP) models.

    Args:
        inputs: Dictionary of model inputs from prepare_data_for_analysis()
        model: PyMC model context
        constraint: Type of constraint for house effects:
            - "zero_median": Median of ALL house effects is zero (default)
            - "zero_sum": Sum/mean of included house effects is zero
            - "none": No constraint (used with anchored models)
    """

    house_effect_sigma = inputs.get("house_effect_sigma", 5.0)
    with model:
        if inputs["right_anchor"] is not None or inputs["left_anchor"] is not None:
            # all house effects are unconstrained, used in anchored models
            house_effects = pm.Normal(
                "house_effects", sigma=house_effect_sigma, shape=inputs["n_firms"]
            )
        elif constraint == "zero_median":
            # Median of ALL house effects is zero
            raw_he = pm.Normal(
                "raw_he",
                sigma=house_effect_sigma,
                shape=inputs["n_firms"],
            )
            house_effects = pm.Deterministic(
                "house_effects", _zero_median_transform(raw_he)
            )
        elif constraint == "zero_sum":
            # Original approach: sum-to-zero for included, unconstrained for excluded
            if len(inputs["he_sum_exclusions"]) > 0:
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
                house_effects = pm.ZeroSumNormal(
                    "house_effects", sigma=house_effect_sigma, shape=inputs["n_firms"]
                )
        else:
            # No constraint
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
    subtracting the mean for the series).

    Args:
        inputs: Dictionary of model inputs from prepare_data_for_analysis()
        **kwargs: Additional arguments including:
            - constraint: House effect constraint type ("zero_median", "zero_sum", "none")
            - innovation: GRW innovation parameter
            - likelihood: Likelihood type ("Normal" or "StudentT")
    """

    constraint = kwargs.pop("constraint", "zero_median")
    model = pm.Model()
    voting_intention = temporal_model(inputs, model, **kwargs)
    house_effects = house_effects_model(inputs, model, constraint=constraint)
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
    using a Gaussian Process (GP).

    Args:
        inputs: Dictionary of model inputs from prepare_data_for_analysis()
        **kwargs: Additional arguments including:
            - constraint: House effect constraint type ("zero_median", "zero_sum", "none")
            - length_scale, eta: GP prior parameters
            - likelihood: Likelihood type ("Normal" or "StudentT")
    """

    constraint = kwargs.pop("constraint", "zero_median")
    model = pm.Model()
    voting_intention = gp_prior(inputs, model, **kwargs)
    house_effects = house_effects_model(inputs, model, constraint=constraint)
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


# --- Residual diagnostics for methodology changes
MIN_POLLS_FOR_RESIDUAL_CHECK = 5
RECENT_POLL_COUNT = 5  # number of most recent polls to highlight
SIGMA_MULTIPLIER = 3  # number of standard deviations for outlier detection
OUTLIER_PCT_THRESHOLD = 0.03  # flag if more than 3% outside (expect ~1% at 3σ)
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold for statistical tests


def _plot_residuals(
    firm_name: str,
    firm_days: np.ndarray,
    firm_residuals: np.ndarray,
    sigma_mean: float,
    day_zero: pd.Period,
    column: str,
    issues: list[str],
    **kwargs,
) -> None:
    """Plot residuals over time with ±Nσ shading for a single pollster.

    Args:
        firm_name: Name of the pollster
        firm_days: Array of day numbers for this pollster's polls
        firm_residuals: Array of residuals for this pollster
        sigma_mean: Model's sigma_likelihood (observation noise)
        day_zero: The reference date (day 0)
        column: The column being analysed (for title)
        issues: List of issues detected for this pollster
        show: If True, display the plot
        **kwargs: Additional arguments passed to mg.finalise_plot
    """
    import matplotlib.pyplot as plt
    import mgplot as mg

    # Convert days to Period then to ordinals for matplotlib compatibility
    periods = [day_zero + int(d) for d in firm_days]
    x_ordinals = [p.ordinal for p in periods]

    _, ax = plt.subplots(figsize=(9.0, 4.5))

    # Shade ±Nσ region
    sigma_band = SIGMA_MULTIPLIER * sigma_mean
    ax.axhspan(
        -sigma_band, sigma_band, alpha=0.2, color="green", label=f"±{SIGMA_MULTIPLIER}σ"
    )

    # Plot zero line
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)

    # Plot residuals
    ax.scatter(x_ordinals, firm_residuals, s=50, alpha=0.7, zorder=5)

    # Mark outliers
    outliers = np.abs(firm_residuals) > sigma_band
    if outliers.any():
        outlier_x = [x for x, o in zip(x_ordinals, outliers) if o]
        outlier_resids = firm_residuals[outliers]
        ax.scatter(
            outlier_x,
            outlier_resids,
            s=80,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label=f"Outside ±{SIGMA_MULTIPLIER}σ",
            zorder=6,
        )

    # Add trend line using mg.line_plot (also sets x-axis date labels)
    z = np.polyfit(firm_days, firm_residuals, 1)
    poly = np.poly1d(z)
    trend_values = poly(firm_days)
    trend_series = pd.Series(trend_values, index=pd.PeriodIndex(periods, freq="D"))
    trend_series.name = "Trend"
    ax = mg.line_plot(trend_series, ax=ax, color="red", style="--", alpha=0.5)

    # Build lfooter from issues
    lfooter_text = "Issues: " + "; ".join(issues) if issues else ""

    defaults = {
        "title": f"Residuals: {firm_name} - {column}",
        "xlabel": None,
        "ylabel": "Residual (percentage points)",
        "legend": {"loc": "best", "fontsize": "x-small"},
        "lfooter": lfooter_text,
        "show": False,
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)

    mg.finalise_plot(ax, **kwargs)


def check_residuals(
    inputs: dict[str, Any],
    trace: az.InferenceData,
    verbose: bool = True,
    plot_suspects: bool = True,
    show: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Check residuals by pollster to detect potential methodology changes.

    For each pollster with at least MIN_POLLS_FOR_RESIDUAL_CHECK polls:
    - Calculates residuals: observed - (voting_intention + house_effect)
    - Checks if residuals are within ±Nσ (N=SIGMA_MULTIPLIER, using model's sigma_likelihood)
    - Tests for heteroskedasticity (variance changing over time)
    - Uses t-test to check for significant mean shift between first/second half
    - Flags recent polls that are outliers

    Args:
        inputs: The prepared data dict from prepare_data_for_analysis()
        trace: InferenceData from model fitting
        verbose: If True, print diagnostic summary
        plot_suspects: If True, plot residuals for suspect pollsters
        show: If True, display the plots
        **kwargs: Additional arguments passed to mg.finalise_plot for residual plots

    Returns:
        DataFrame with residual diagnostics per pollster, including:
        - n_polls: number of polls from this pollster
        - n_outside_3sd: count of residuals outside ±Nσ
        - pct_outside_3sd: percentage outside
        - het_pvalue: p-value for heteroskedasticity test
        - recent_3s: count of recent polls outside ±3σ
        - recent_2s: count of recent polls outside ±2σ
        - mean_shift: difference in mean residual (2nd half - 1st half)
        - mean_shift_pvalue: p-value from t-test comparing halves
        - suspect: True if methodology change may be indicated
    """
    from scipy import stats

    # Extract posterior means
    vi_posterior = trace.posterior["voting_intention"]
    vi_mean = vi_posterior.mean(dim=["chain", "draw"]).values

    he_posterior = trace.posterior["house_effects"]
    he_mean = he_posterior.mean(dim=["chain", "draw"]).values

    sigma_posterior = trace.posterior["sigma_likelihood"]
    sigma_mean = float(sigma_posterior.mean(dim=["chain", "draw"]).values)

    # Determine if this is a GRW model (voting_intention indexed by day)
    # or GP model (voting_intention indexed by poll)
    is_grw = len(vi_mean) == inputs["n_days"] + 1

    # Calculate residuals for each poll
    poll_days = inputs["poll_day"].values
    poll_firms = inputs["poll_firm_number"].values
    observed = inputs["zero_centered_y"].values

    if is_grw:
        expected = vi_mean[poll_days] + he_mean[poll_firms]
    else:
        expected = vi_mean + he_mean[poll_firms]

    residuals = observed - expected

    # Build results by pollster
    results = []
    suspect_data = []  # Store data for plotting suspects

    for firm_name in inputs["firm_list"]:
        firm_num = inputs["firm_map"][firm_name]
        mask = poll_firms == firm_num
        n_polls = mask.sum()

        if n_polls < MIN_POLLS_FOR_RESIDUAL_CHECK:
            continue

        firm_residuals = residuals[mask]
        firm_days = poll_days[mask]

        # Check proportion outside ±Nσ
        sigma_band = SIGMA_MULTIPLIER * sigma_mean
        outside_band = np.abs(firm_residuals) > sigma_band
        n_outside = outside_band.sum()
        pct_outside = n_outside / n_polls

        # Test for heteroskedasticity using Breusch-Pagan-like approach
        # Regress squared residuals on time (day number)
        squared_resid = firm_residuals**2
        slope, intercept, r_value, het_pvalue, std_err = stats.linregress(
            firm_days, squared_resid
        )

        # Check recent polls for outliers
        # Flag if 1+ at 3σ or 2+ at 2σ
        firm_indices = np.where(mask)[0]
        recent_indices = firm_indices[-RECENT_POLL_COUNT:]
        recent_residuals = residuals[recent_indices]
        recent_outside_3s = np.abs(recent_residuals) > 3 * sigma_mean
        recent_outside_2s = np.abs(recent_residuals) > 2 * sigma_mean
        n_recent_3s = recent_outside_3s.sum()
        n_recent_2s = recent_outside_2s.sum()
        recent_outlier_flag = (n_recent_3s >= 1) or (n_recent_2s >= 2)

        # T-test for mean shift between first and second half
        half = len(firm_residuals) // 2
        first_half = firm_residuals[:half]
        second_half = firm_residuals[half:]
        mean_shift = second_half.mean() - first_half.mean()

        # Use Welch's t-test (doesn't assume equal variances)
        t_stat, mean_shift_pvalue = stats.ttest_ind(
            first_half, second_half, equal_var=False
        )

        # Determine if suspect and collect issues
        # Criteria:
        # - More than OUTLIER_PCT_THRESHOLD outside Nσ
        # - Heteroskedasticity p-value < SIGNIFICANCE_LEVEL
        # - More than 1 recent outlier (out of RECENT_POLL_COUNT)
        # - Mean shift t-test p-value < SIGNIFICANCE_LEVEL
        issues: list[str] = []
        if pct_outside > OUTLIER_PCT_THRESHOLD:
            issues.append(f"{pct_outside:.0%} outside ±{SIGMA_MULTIPLIER}σ")
        if het_pvalue < SIGNIFICANCE_LEVEL:
            direction = "increasing" if slope > 0 else "decreasing"
            issues.append(f"variance {direction} (p={het_pvalue:.3f})")
        if recent_outlier_flag:
            issues.append(f"recent: {n_recent_3s}@3σ, {n_recent_2s}@2σ")
        if mean_shift_pvalue < SIGNIFICANCE_LEVEL:
            issues.append(f"mean shift {mean_shift:+.2f} (p={mean_shift_pvalue:.3f})")

        suspect = len(issues) > 0

        results.append(
            {
                "pollster": firm_name,
                "n_polls": n_polls,
                "n_outside_3sd": n_outside,
                "pct_outside_3sd": pct_outside,
                "het_pvalue": het_pvalue,
                "het_slope": slope,
                "recent_3s": n_recent_3s,
                "recent_2s": n_recent_2s,
                "mean_shift": mean_shift,
                "mean_shift_pvalue": mean_shift_pvalue,
                "suspect": suspect,
            }
        )

        if suspect:
            suspect_data.append(
                {
                    "firm_name": firm_name,
                    "firm_days": firm_days,
                    "firm_residuals": firm_residuals,
                    "issues": issues,
                }
            )

    results_df = pd.DataFrame(results)

    if verbose and len(results_df) > 0:
        print("\n=== Residual Diagnostics by Pollster ===")
        print(f"Model sigma_likelihood: {sigma_mean:.2f}")
        print(f"Minimum polls required: {MIN_POLLS_FOR_RESIDUAL_CHECK}\n")

        for _, row in results_df.iterrows():
            flag = " ⚠️ SUSPECT" if row["suspect"] else ""
            print(f"{row['pollster']} (n={row['n_polls']}){flag}")
            print(
                f"  Outside ±{SIGMA_MULTIPLIER}σ: {row['n_outside_3sd']}/{row['n_polls']} "
                f"({row['pct_outside_3sd']:.1%})"
            )
            print(f"  Heteroskedasticity p-value: {row['het_pvalue']:.3f}", end="")
            if row["het_pvalue"] < SIGNIFICANCE_LEVEL:
                direction = "increasing" if row["het_slope"] > 0 else "decreasing"
                print(f" (variance {direction})")
            else:
                print(" (homoskedastic)")
            print(
                f"  Recent outliers: {row['recent_3s']}@3σ, {row['recent_2s']}@2σ (of {RECENT_POLL_COUNT})"
            )
            print(
                f"  Mean shift: {row['mean_shift']:+.2f} "
                f"(t-test p={row['mean_shift_pvalue']:.3f})"
            )
            print()

        suspects = results_df[results_df["suspect"]]
        if len(suspects) > 0:
            print(
                f"⚠️  {len(suspects)} pollster(s) flagged for potential "
                "methodology issues:"
            )
            for name in suspects["pollster"]:
                print(f"   - {name}")
        else:
            print("✓ No pollsters flagged for methodology concerns.")

    # Plot residuals for suspect pollsters
    if plot_suspects and suspect_data:
        column = inputs.get("column", "Unknown")
        for data in suspect_data:
            _plot_residuals(
                firm_name=data["firm_name"],
                firm_days=data["firm_days"],
                firm_residuals=data["firm_residuals"],
                sigma_mean=sigma_mean,
                day_zero=inputs["day_zero"],
                column=column,
                issues=data["issues"],
                show=show,
                **kwargs,
            )

    return results_df
