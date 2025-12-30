"""Bar chart plots for scalar posterior distributions."""

import math

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import pandas as pd

from extraction import get_scalar_var, get_scalar_var_names


def _auto_scale(samples: pd.Series, median: float) -> tuple[pd.Series, int]:
    """Scale samples for better visualization when values are large."""
    threshold = 1.3
    if abs(median) <= threshold:
        return samples, 1
    scale = 10 ** math.floor(math.log10(abs(median * 10)))
    return samples / scale, max(int(scale), 1)


def plot_posteriors_bar(
    trace: az.InferenceData,
    series_name: str = "Model",
    palette: str = "Blues",
    figsize: tuple[float, float] = (9.0, 2.0),
    **kwargs,
) -> None:
    """Plot horizontal bar chart of scalar posterior distributions.

    Args:
        trace: InferenceData from PyMC sampling
        series_name: Name of the data series (e.g., "2PP vote ALP")
        palette: Matplotlib colormap name (e.g., "Blues", "Reds", "Greens")
        figsize: Figure size as (width, height) tuple
        **kwargs: Additional arguments passed to mg.finalise_plot
    """
    scalar_vars = get_scalar_var_names(trace)

    posteriors = {}
    labels = {}
    all_significant_99 = True
    all_significant_95 = True

    for var in scalar_vars:
        samples = get_scalar_var(var, trace)
        median = samples.quantile(0.5)

        if median < 0:
            if samples.quantile(0.99) >= 0:
                all_significant_99 = False
            if samples.quantile(0.95) >= 0:
                all_significant_95 = False
        else:
            if samples.quantile(0.01) <= 0:
                all_significant_99 = False
            if samples.quantile(0.05) <= 0:
                all_significant_95 = False

        scaled_samples, scale = _auto_scale(samples, median)
        if scale != 1:
            posteriors[var] = scaled_samples
            labels[var] = f"{var}/{scale}"
        else:
            posteriors[var] = samples
            labels[var] = var

    cuts = [2.5, 16]
    cmap = plt.get_cmap(palette)
    color_fracs = [0.4, 0.7]

    _, ax = plt.subplots(figsize=figsize)

    y_positions = range(len(scalar_vars))
    bar_height = 0.7

    sorted_vars = sorted(scalar_vars)
    for i, var in enumerate(sorted_vars):
        samples = posteriors[var]

        for j, p in enumerate(cuts):
            quants = (p, 100 - p)
            lower = samples.quantile(quants[0] / 100.0)
            upper = samples.quantile(quants[1] / 100.0)
            height = bar_height * (1 - j * 0.25)

            ax.barh(
                i,
                width=upper - lower,
                left=lower,
                height=height,
                color=cmap(color_fracs[j]),
                alpha=0.7,
                label=f"{quants[1] - quants[0]:.0f}% HDI" if i == 0 else "_",
                zorder=j + 1,
            )

        median = samples.quantile(0.5)
        ax.vlines(
            median,
            i - bar_height / 2,
            i + bar_height / 2,
            color="black",
            linestyle="-",
            linewidth=1,
            zorder=10,
            label="Median" if i == 0 else "_",
        )
        ax.text(
            median,
            i,
            f"{median:.3f}",
            ha="center",
            va="center",
            fontsize="small",
            fontweight="bold",
            color="white",
            zorder=15,
        )

    ax.axvline(x=0, color="darkred", linestyle="-", linewidth=1.5, zorder=15)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([labels[var] for var in sorted_vars])
    ax.invert_yaxis()

    lfooter = ""
    if any(
        scale != 1
        for _, scale in [
            _auto_scale(posteriors[v], posteriors[v].quantile(0.5)) for v in sorted_vars
        ]
    ):
        lfooter = "Some variables have been scaled (as indicated). "
    if all_significant_99:
        lfooter += "All coefficients are different from zero (>99% probability)."
    elif all_significant_95:
        lfooter += "All coefficients are different from zero (>95% probability)."

    defaults = {
        "title": f"Scalar Posteriors: {series_name}",
        "xlabel": "Parameter value",
        "legend": {"loc": "best", "fontsize": "x-small"},
        "lfooter": lfooter if lfooter else None,
        "figsize": figsize,
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs)
