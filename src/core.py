"""Core functions for Poisson regression analysis of public health policy."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper


def generate_synthetic_data(
    n_samples: int = 2000,
    intercept: float = 0.5,
    intervention_effect: float = -0.30,
    age_effect: float = 0.15,
    income_effect: float = -0.10,
    urban_effect: float = 0.20,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate hospital visits as a function of a public health intervention.

    `hospital_visits` is drawn from a Poisson distribution whose rate depends
    on whether the individual received the intervention, their age (decades),
    income (tens of thousands), and whether they live in an urban area.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 85, n_samples)
    income = rng.uniform(20, 120, n_samples)
    urban = rng.integers(0, 2, n_samples)
    intervention = rng.integers(0, 2, n_samples)

    log_rate = (
        intercept
        + intervention_effect * intervention
        + age_effect * (age / 10)
        + income_effect * (income / 10)
        + urban_effect * urban
    )
    hospital_visits = rng.poisson(np.exp(log_rate))

    return pd.DataFrame(
        {
            "hospital_visits": hospital_visits,
            "intervention": intervention,
            "age": age,
            "income": income,
            "urban": urban,
        }
    )


def fit_poisson_regression(df: pd.DataFrame) -> GLMResultsWrapper:
    """Fit a Poisson regression of hospital visits on policy and demographic variables."""
    formula = "hospital_visits ~ intervention + age + income + urban"
    return smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()


def add_predictions(df: pd.DataFrame, model: GLMResultsWrapper) -> pd.DataFrame:
    """Add predicted values and residuals from a fitted model to a copy of df."""
    out = df.copy()
    out["predicted"] = model.predict()
    out["residuals"] = out["hospital_visits"] - out["predicted"]
    return out


def plot_hospital_visits_distribution(
    df: pd.DataFrame, output_path: Path, figsize: tuple[int, int] = (6, 5), plot: bool = False
) -> None:
    """Plot a histogram of the hospital visits distribution."""
    if not plot:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df["hospital_visits"], bins=20, color="#4A90A4", edgecolor="none")
    ax.set_title("Distribution of Hospital Visits")
    ax.set_xlabel("Number of Visits")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame, output_path: Path, figsize: tuple[int, int] = (6, 5), plot: bool = False
) -> None:
    """Plot a correlation matrix heatmap for the simulated variables."""
    if not plot:
        return
    import matplotlib.pyplot as plt

    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Correlation Matrix")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_residuals_vs_predicted(
    df: pd.DataFrame, output_path: Path, figsize: tuple[int, int] = (6, 5), plot: bool = False
) -> None:
    """Plot residuals against predicted values for model diagnostics."""
    if not plot:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df["predicted"], df["residuals"], color="#4A90A4", alpha=0.5, edgecolor="none")
    ax.axhline(0, color="#D4A574", linestyle="--")
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
