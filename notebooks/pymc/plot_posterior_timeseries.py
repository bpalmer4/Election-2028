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

# Color fractions for colormap shades (matching house effects chart)
COLOR_FRACS = (0.3, 0.5, 0.7)
FIXED_ALPHA = 0.4


def _rgba_to_hex(rgba: tuple) -> str:
    """Convert RGBA tuple to hex string for compatibility with mgplot."""
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


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
    palette: str = "Blues",
    start: pd.Period | None = None,
    cuts: Sequence[float] = (0.005, 0.025, 0.16),
    color_fracs: Sequence[float] = COLOR_FRACS,
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
        palette: Matplotlib colormap name for consistent coloring
        start: Start period for filtering
        cuts: Quantile cuts for credible intervals
        color_fracs: Color fractions from colormap for each credible interval
        finalise: If True, call finalise_plot and return None.
            If False, return Axes for composition.
        **finalise_kwargs: Passed to mg.finalise_plot (title, lfooter, rfooter, etc.)

    Returns:
        Axes if finalise=False, None if finalise=True.

    """
    if len(cuts) != len(color_fracs):
        raise ValueError("Cuts and color_fracs must have the same length")

    cmap = plt.get_cmap(palette)

    if data is not None:
        samples = data
    elif trace is not None and var is not None and index is not None:
        samples = get_vector_var(var, trace)
        samples.index = index
    else:
        raise ValueError("Must provide either (trace, var, index) or data")

    if start is not None:
        samples = samples[samples.index >= start]

    # Get colors from colormap (convert to hex for mgplot compatibility)
    main_color = _rgba_to_hex(cmap(0.7))  # For points, reference line, etc.
    median_color = _rgba_to_hex(cmap(0.9))  # Darker shade for median line

    # Create axes and plot raw polls first (so credible intervals overlay them)
    _, ax = plt.subplots()
    if poll_data is not None and poll_column is not None:
        _plot_raw_polls(ax, poll_data, poll_column, pollster_column, main_color)

    # Add election result reference line
    if election_result is not None:
        ax.axhline(
            election_result,
            color=main_color,
            linestyle="--",
            linewidth=1.5,
            label=f"2025 Result: {election_result:.1f}%",
            zorder=4,
        )

    for cut, color_frac in zip(cuts, color_fracs):
        if not (0 < cut < 0.5):
            raise ValueError("Cuts must be between 0 and 0.5")

        lower = samples.quantile(q=cut, axis=1)
        upper = samples.quantile(q=1 - cut, axis=1)
        band = pd.DataFrame({"lower": lower, "upper": upper}, index=samples.index)
        band_color = _rgba_to_hex(cmap(color_frac))
        ax = mg.fill_between_plot(
            band,
            ax=ax,
            color=band_color,
            alpha=FIXED_ALPHA,
            label=f"{legend_stem} {int((1 - 2 * cut) * 100)}% Credible Interval",
        )

    median = samples.quantile(q=0.5, axis=1)
    median.name = f"{legend_stem} Median"
    ax = mg.line_plot(median, ax=ax, color=median_color, width=1, annotate=True)

    if finalise and ax is not None:
        mg.finalise_plot(ax, **finalise_kwargs)
        return None

    return ax
