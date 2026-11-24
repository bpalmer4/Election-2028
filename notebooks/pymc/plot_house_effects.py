"""Bar chart plots for posterior distributions."""

import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
import xarray as xr

from extraction import get_house_effects_var


def plot_house_effects_bar(
    trace: xr.DataTree,
    firm_map: dict[int, str],
    model_name: str = "Model",
    poll_counts: dict[str, int] | None = None,
    **kwargs,
) -> pd.Series:
    """Plot horizontal bar chart of house effects posteriors.

    Args:
        trace: ArviZ posterior DataTree object
        firm_map: Mapping from index to pollster name (inputs["back_firm_map"])
        model_name: Name for the model (used in plot title/footer)
        poll_counts: Optional mapping from pollster name to poll count
        **kwargs: Additional arguments passed to mg.finalise_plot

    Returns:
        Series of median house effects by pollster name
    """
    df = get_house_effects_var(trace).rename(index=firm_map)

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


def plot_house_effects_eras(
    trace: xr.DataTree,
    inputs: dict,
    model_name: str = "Model",
    **kwargs,
) -> pd.DataFrame:
    """Plot each pollster's house effect before and after the era break.

    One row per pollster: the pre-break estimate (lighter) above the
    post-break estimate (darker), each with a median dot, 50% (thick) and
    95% (thin) intervals, joined by a dotted line when both exist.
    Requires inputs from prepare_data_for_analysis(..., era_break=...).

    Args:
        trace: ArviZ posterior DataTree object
        inputs: Prepared data dict (needs era_break, brand_list, unit_*)
        model_name: Name for the model (used in plot title/footer)
        **kwargs: Additional arguments passed to mg.finalise_plot
            (palette is consumed here)

    Returns:
        DataFrame indexed by pollster: pre, post, shift (post - pre),
        n_pre, n_post. Missing eras are NaN / 0.
    """
    era_break = inputs.get("era_break")
    if era_break is None:
        raise ValueError("plot_house_effects_eras needs inputs with an era_break.")
    samples = get_house_effects_var(trace)

    # posterior summaries per (pollster, era) unit
    by_brand: dict[str, dict[int, dict[str, float]]] = {}
    for unit, label in inputs["back_firm_map"].items():
        s = samples.loc[unit]
        brand = inputs["brand_list"][int(inputs["unit_brand"][unit])]
        by_brand.setdefault(brand, {})[int(inputs["unit_era"][unit])] = {
            "median": float(s.quantile(0.5)),
            "lo95": float(s.quantile(0.025)),
            "hi95": float(s.quantile(0.975)),
            "lo50": float(s.quantile(0.25)),
            "hi50": float(s.quantile(0.75)),
            "n": float(inputs["poll_counts"][label]),
        }

    # order by post-break median; pre-only pollsters at the bottom
    def _sort_key(brand: str) -> tuple[bool, float]:
        eras = by_brand[brand]
        return (1 not in eras, eras[1 if 1 in eras else 0]["median"])

    order = sorted(by_brand, key=_sort_key)

    palette = kwargs.pop("palette", "Blues")
    cmap = plt.get_cmap(palette)
    era_style = {
        0: (cmap(0.45), -0.17, f"Before {era_break:%-d %b %Y}"),
        1: (cmap(0.9), 0.17, f"From {era_break:%-d %b %Y}"),
    }

    _, ax = plt.subplots(figsize=(10, len(order) * 0.6 + 1))
    # legend proxies in era order (the first pollster drawn may be post-only)
    for color, _dy, legend_label in era_style.values():
        ax.plot([], [], color=color, linewidth=1.5, label=legend_label)
    # value labels sit in a column just right of the axes, clear of the data
    label_transform = ax.get_yaxis_transform()
    for i, brand in enumerate(order):
        eras = by_brand[brand]
        if len(eras) == len(era_style):
            ax.plot(
                [eras[0]["median"], eras[1]["median"]],
                [i + era_style[0][1], i + era_style[1][1]],
                color="grey",
                linestyle=":",
                linewidth=1,
                zorder=1,
            )
        for era, r in eras.items():
            color, dy, _legend_label = era_style[era]
            y = i + dy
            ax.hlines(y, r["lo95"], r["hi95"], color=color, linewidth=1.5, zorder=2)
            ax.hlines(y, r["lo50"], r["hi50"], color=color, linewidth=6, zorder=3)
            ax.plot(r["median"], y, "o", color="white", markeredgecolor=color, zorder=4)
            ax.text(
                1.01,
                y,
                f"{r['median']:+.1f}",
                transform=label_transform,
                ha="left",
                va="center",
                fontsize=8,
                color=color,
                clip_on=False,
            )

    def _n(brand: str, era: int) -> int:
        return int(by_brand[brand][era]["n"]) if era in by_brand[brand] else 0

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{b} (n={_n(b, 0)}/{_n(b, 1)})" for b in order], fontsize=9)
    ax.invert_yaxis()

    defaults = {
        "title": f"House Effects Before/After Break: {model_name}",
        "xlabel": "Relative effect (percentage points)",
        "axvline": {"x": 0, "color": "black", "linestyle": "-", "linewidth": 0.5},
        "legend": {"loc": "best", "fontsize": "x-small"},
        "rfooter": model_name,
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    mg.finalise_plot(ax, **kwargs)

    def _median(brand: str, era: int) -> float:
        return by_brand[brand][era]["median"] if era in by_brand[brand] else np.nan

    summary = pd.DataFrame(
        {
            "pre": [_median(b, 0) for b in order],
            "post": [_median(b, 1) for b in order],
            "n_pre": [_n(b, 0) for b in order],
            "n_post": [_n(b, 1) for b in order],
        },
        index=pd.Index(order, name="pollster"),
    )
    summary.insert(2, "shift", summary["post"] - summary["pre"])
    return summary


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

        median = float(samples.quantile(0.5))
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
    ax.axhline(
        median_position,
        color="darkgrey",
        linestyle="--",
        linewidth=1.5,
        zorder=10,
        label="Median House",
    )

    kwargs.setdefault(
        "axvline",
        {"x": 0, "color": "black", "linestyle": "-", "linewidth": 0.5, "zorder": 99},
    )
    kwargs.setdefault("title", "Posterior Distributions")
    kwargs.setdefault("xlabel", "Value")
    kwargs.setdefault("legend", {"loc": "upper right", "fontsize": "x-small"})
    kwargs.setdefault("rfooter", model_name)

    mg.finalise_plot(ax, **kwargs)
