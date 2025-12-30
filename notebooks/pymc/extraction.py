"""Extract variables from PyMC traces."""

import arviz as az
import pandas as pd


def get_vector_var(var_name: str, trace: az.InferenceData) -> pd.DataFrame:
    """Extract chains/draws for a vector variable.

    Returns DataFrame with rows=vector elements, columns=samples.
    """
    # Stack chains and draws into a single sample dimension
    data = trace.posterior[var_name].stack(sample=("chain", "draw"))
    # Convert to DataFrame: rows are vector elements, columns are samples
    return pd.DataFrame(data.values, index=range(data.shape[0]))


def get_scalar_var(var_name: str, trace: az.InferenceData) -> pd.Series:
    """Extract chains/draws for a scalar variable as a Series."""
    data = trace.posterior[var_name].stack(sample=("chain", "draw"))
    return pd.Series(data.values.flatten())


def get_scalar_var_names(trace: az.InferenceData) -> list[str]:
    """Get names of scalar (non-vector) variables from trace."""
    scalar_vars = []
    for var_name in trace.posterior.data_vars:
        var = trace.posterior[var_name]
        # Scalar if only dimensions are chain and draw
        if set(var.dims) == {"chain", "draw"}:
            scalar_vars.append(var_name)
    return scalar_vars


def get_house_effects_var(trace: az.InferenceData) -> pd.DataFrame:
    """Extract house effects from trace, handling both combined and split cases.

    Handles:
    - Single 'house_effects' variable
    - Split 'zero_sum_he' + optional 'unconstrained_he' variables

    Returns DataFrame with rows=pollsters, columns=samples.
    Raises ValueError if no house effects variables found.
    """
    posterior_vars = list(trace.posterior.data_vars)

    if "house_effects" in posterior_vars:
        return get_vector_var("house_effects", trace)

    if "zero_sum_he" in posterior_vars:
        df_zero = get_vector_var("zero_sum_he", trace)
        if "unconstrained_he" in posterior_vars:
            df_unconst = get_vector_var("unconstrained_he", trace)
            return pd.concat([df_zero, df_unconst], ignore_index=True)
        return df_zero

    raise ValueError("No house effects variables found in trace")
