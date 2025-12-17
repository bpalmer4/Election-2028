"""Unified posterior time series plotting with finalisation."""

from collections.abc import Sequence
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import pandas as pd
from matplotlib.axes import Axes

from extraction import get_vector_var

# Symbols for different pollsters
POLLSTER_SYMBOLS = list("os^v<>Dp*hH+x1234")


def _plot_raw_polls(
    ax: Axes,
    poll_data: pd.DataFrame,
    poll_column: str,
    pollster_column: str = "Brand",
    color: str = "blue",
) -> None:
    """Plot raw poll data points with different symbols per pollster."""
    for i, pollster in enumerate(sorted(poll_data[pollster_column].unique())):
        selection = poll_data[poll_data[pollster_column] == pollster][poll_column]
        if selection.empty or selection.isna().all():
            continue
        # Use ordinals for x-axis to match mgplot's internal representation
        x = [p.ordinal for p in selection.index]
        ax.scatter(
            x,
            selection.to_numpy(),
            color=color,
            s=25,
            marker=POLLSTER_SYMBOLS[i % len(POLLSTER_SYMBOLS)],
            alpha=0.6,
            label=pollster,
            zorder=5,
        )


def plot_posterior_timeseries(
    trace: az.InferenceData | None = None,
    var: str | None = None,
    index: pd.PeriodIndex | None = None,
    data: pd.DataFrame | None = None,
    poll_data: pd.DataFrame | None = None,
    poll_column: str | None = None,
    pollster_column: str = "Brand",
    election_result: float | None = None,
    legend_stem: str = "Model Estimate",
    color: str = "blue",
    start: pd.Period | None = None,
    cuts: Sequence[float] = (0.005, 0.025, 0.16),
    alphas: Sequence[float] = (0.1, 0.2, 0.3),
    finalise: bool = True,
    **finalise_kwargs: Any,
) -> Axes | None:
    """Plot posterior time series with credible intervals.

    Supports two modes:
    1. Direct from trace: provide (trace, var, index)
    2. Pre-computed DataFrame: provide data

    Args:
        trace: ArviZ InferenceData object
        var: Variable name to extract from trace
        index: PeriodIndex for the time series
        data: Pre-computed DataFrame of samples (time × draws)
        poll_data: Optional DataFrame with raw poll data to plot as points
        poll_column: Column name for poll values in poll_data
        pollster_column: Column name for pollster/brand in poll_data (default "Brand")
        election_result: Optional previous election result to plot as horizontal line
        legend_stem: Stem for legend labels
        color: Color for the plot
        start: Start period for filtering
        cuts: Quantile cuts for credible intervals
        alphas: Alpha values for credible interval bands
        finalise: If True, call finalise_plot and return None.
            If False, return Axes for composition.
        **finalise_kwargs: Passed to mg.finalise_plot (title, lfooter, rfooter, etc.)

    Returns:
        Axes if finalise=False, None if finalise=True.

    """
    if len(cuts) != len(alphas):
        raise ValueError("Cuts and alphas must have the same length")

    if data is not None:
        samples = data
    elif trace is not None and var is not None and index is not None:
        samples = get_vector_var(var, trace)
        samples.index = index
    else:
        raise ValueError("Must provide either (trace, var, index) or data")

    if start is not None:
        samples = samples[samples.index >= start]

    # Create axes and plot raw polls first (so credible intervals overlay them)
    _, ax = plt.subplots()
    if poll_data is not None and poll_column is not None:
        _plot_raw_polls(ax, poll_data, poll_column, pollster_column, color)

    # Add election result reference line
    if election_result is not None:
        ax.axhline(
            election_result,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"2025 Result: {election_result:.1f}%",
            zorder=4,
        )

    for cut, alpha in zip(cuts, alphas):
        if not (0 < cut < 0.5):
            raise ValueError("Cuts must be between 0 and 0.5")

        lower = samples.quantile(q=cut, axis=1)
        upper = samples.quantile(q=1 - cut, axis=1)
        band = pd.DataFrame({"lower": lower, "upper": upper}, index=samples.index)
        ax = mg.fill_between_plot(
            band,
            ax=ax,
            color=color,
            alpha=alpha,
            label=f"{legend_stem} {int((1 - 2 * cut) * 100)}% Credible Interval",
        )

    median = samples.quantile(q=0.5, axis=1)
    median.name = f"{legend_stem} Median"
    ax = mg.line_plot(median, ax=ax, color=color, width=1, annotate=True)

    if finalise and ax is not None:
        mg.finalise_plot(ax, **finalise_kwargs)
        return None

    return ax
