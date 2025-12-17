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
