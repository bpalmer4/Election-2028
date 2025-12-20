"""Bar chart plots for posterior distributions."""

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import pandas as pd

from extraction import get_vector_var


def plot_house_effects_bar(
    trace: az.InferenceData,
    firm_map: dict[int, str],
    model_name: str = "Model",
    poll_counts: dict[str, int] | None = None,
    **kwargs,
) -> pd.Series:
    """Plot horizontal bar chart of house effects posteriors.

    Args:
        trace: ArviZ InferenceData object
        firm_map: Mapping from index to pollster name (inputs["back_firm_map"])
        model_name: Name for the model (used in plot title/footer)
        poll_counts: Optional mapping from pollster name to poll count
        **kwargs: Additional arguments passed to mg.finalise_plot

    Returns:
        Series of median house effects by pollster name
    """
    # Extract house effects - handle both combined and split variable cases
    posterior_vars = list(trace.posterior.data_vars)

    if "house_effects" in posterior_vars:
        df = get_vector_var("house_effects", trace)
    elif "zero_sum_he" in posterior_vars:
        # Combine zero_sum_he and unconstrained_he
        df_zero = get_vector_var("zero_sum_he", trace)
        if "unconstrained_he" in posterior_vars:
            df_unconst = get_vector_var("unconstrained_he", trace)
            # Concatenate: inclusions first (zero_sum), then exclusions (unconstrained)
            df = pd.concat([df_zero, df_unconst], ignore_index=True)
        else:
            df = df_zero
    else:
        raise ValueError("No house effects variables found in trace")

    df = df.rename(index=firm_map)

    medians = df.quantile(0.5, axis=1).sort_values()
    df = df.reindex(medians.index)

    # Create labels with poll counts if provided
    if poll_counts:
        posteriors = {
            f"{name} (n={poll_counts.get(name, '?')})": df.loc[name]
            for name in df.index
        }
    else:
        posteriors = {name: df.loc[name] for name in df.index}

    defaults = {
        "title": f"House Effects: {model_name}",
        "xlabel": "Relative effect (percentage points)",
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)

    _plot_bar_chart(posteriors, model_name, **kwargs)

    return medians


def _plot_bar_chart(
    posteriors: dict[str, pd.Series],
    model_name: str = "Model",
    **kwargs,
) -> None:
    """Plot horizontal bar chart of posteriors."""

    cuts = [2.5, 12.5, 25]
    palette = kwargs.pop("palette", "Blues")
    cmap = plt.get_cmap(palette)
    color_fracs = [0.3, 0.5, 0.7]

    n_vars = len(posteriors)
    _, ax = plt.subplots(figsize=(10, n_vars * 0.5 + 1))

    bar_height = 0.7
    sorted_vars = list(posteriors.keys())

    for i, var in enumerate(sorted_vars):
        samples = posteriors[var]

        for j, p in enumerate(cuts):
            quants = (p, 100 - p)
            lower = samples.quantile(quants[0] / 100.0)
            upper = samples.quantile(quants[1] / 100.0)
            height = bar_height * (1 - j * 0.15)

            ax.barh(
                i,
                width=upper - lower,
                left=lower,
                height=height,
                color=cmap(color_fracs[j]),
                alpha=0.8,
                label=f"{quants[1] - quants[0]:.0f}% HDI" if i == 0 else "_",
                zorder=j + 1,
            )

        median = samples.quantile(0.5)
        ax.text(
            median,
            i,
            f"{median:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=100,
        )

    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(sorted_vars, fontsize=9)
    ax.invert_yaxis()

    # Add horizontal dashed line at median position (zorder between bars and text)
    median_position = (n_vars - 1) / 2
    ax.axhline(median_position, color="darkgrey", linestyle="--", linewidth=1.5, zorder=10, label="Median House")

    kwargs.setdefault("axvline", {"x": 0, "color": "black", "linestyle": "-", "linewidth": 0.5, "zorder": 99})
    kwargs.setdefault("title", "Posterior Distributions")
    kwargs.setdefault("xlabel", "Value")
    kwargs.setdefault("legend", {"loc": "upper right", "fontsize": "x-small"})
    kwargs.setdefault("rfooter", model_name)

    mg.finalise_plot(ax, **kwargs)
