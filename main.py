#!/usr/bin/env python3
"""
Generalized Linear Models for Econometric Analysis of Public Policy

Main entry point: simulates the public health intervention / hospital
visits case study and fits a Poisson regression model.
"""

import argparse
import logging
from pathlib import Path

import yaml

from src.core import (
    add_predictions,
    fit_poisson_regression,
    generate_synthetic_data,
    plot_correlation_heatmap,
    plot_hospital_visits_distribution,
    plot_residuals_vs_predicted,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generalized Linear Models for Econometric Analysis of Public Policy"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config file")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["output"]["figures_dir"])
    )
    output_dir.mkdir(exist_ok=True)

    df = generate_synthetic_data(
        n_samples=config["data"]["n_samples"],
        intercept=config["data"]["intercept"],
        intervention_effect=config["data"]["intervention_effect"],
        age_effect=config["data"]["age_effect"],
        income_effect=config["data"]["income_effect"],
        urban_effect=config["data"]["urban_effect"],
        seed=config["data"]["seed"],
    )

    model = fit_poisson_regression(df)
    logging.info(model.summary().as_text())
    logging.info(f"Deviance: {model.deviance:.2f}")
    logging.info(f"Degrees of Freedom: {model.df_resid:.0f}")

    df = add_predictions(df, model)

    figsize = tuple(config["output"]["figsize"])
    plot_hospital_visits_distribution(df, output_dir / "hospital_visits_distribution.png", figsize)
    plot_correlation_heatmap(df, output_dir / "correlation_matrix.png", figsize)
    plot_residuals_vs_predicted(df, output_dir / "residuals_vs_predicted.png", figsize)

    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
