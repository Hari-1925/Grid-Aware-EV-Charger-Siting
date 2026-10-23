"""Demand scoring and supply gap analysis for EV charger siting.

This module computes multi-criteria demand scores for spatial grid cells,
evaluates unmet charging demand (supply gap), and generates demand heatmaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to the range [0, 1].

    Parameters
    ----------
    series : pd.Series
        Numeric pandas series to normalize.

    Returns
    -------
    pd.Series
        Normalized series with values in [0, 1]. Returns series of 0.0 if
        max == min or all values are NaN.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    s_min = s.min()
    s_max = s.max()
    if s_max > s_min:
        return (s - s_min) / (s_max - s_min)
    return pd.Series(0.0, index=series.index, dtype=float)


def _get_feature_series(
    gdf: gpd.GeoDataFrame,
    norm_col: str,
    raw_col: Optional[str] = None,
    default_val: float = 0.0,
) -> pd.Series:
    """Retrieve normalized feature column or normalize the raw feature column.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing cell attributes.
    norm_col : str
        Expected column name for normalized feature (ending in '_norm').
    raw_col : Optional[str]
        Fallback column name for raw unnormalized feature.
    default_val : float
        Default fill value if neither column exists.

    Returns
    -------
    pd.Series
        Feature series guaranteed to be normalized in [0, 1].
    """
    if norm_col in gdf.columns:
        s = pd.to_numeric(gdf[norm_col], errors="coerce").fillna(default_val)
        return s.clip(0.0, 1.0)

    if raw_col is not None and raw_col in gdf.columns:
        logger.info(
            "Column '%s' not found. Normalizing raw column '%s'.",
            norm_col,
            raw_col,
        )
        return min_max_normalize(gdf[raw_col])

    logger.warning(
        "Neither '%s' nor '%s' found in grid_gdf. Defaulting to %.2f.",
        norm_col,
        raw_col,
        default_val,
    )
    return pd.Series(default_val, index=gdf.index, dtype=float)


def compute_demand_score(
    grid_gdf: gpd.GeoDataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> gpd.GeoDataFrame:
    """Compute weighted EV charging demand score per grid cell.

    Formula:
        demand = w_pop * population_proxy_norm
               + w_poi * (poi_count_norm + poi_diversity_norm) / 2
               + w_traffic * traffic_proxy_norm
               - w_charger_penalty * chargers_in_radius_norm

    Resulting demand score is clipped to [0, 1] and added as 'demand_score'.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Spatial grid cells with normalized feature columns.
    config : Optional[Dict[str, Any]]
        Configuration dictionary containing 'feature_weights'. Defaults are:
        - population: 0.35
        - poi: 0.25
        - traffic: 0.25
        - existing_chargers_penalty: 0.15

    Returns
    -------
    gpd.GeoDataFrame
        Copy of grid_gdf with added 'demand_score' column.
    """
    if not isinstance(grid_gdf, gpd.GeoDataFrame):
        raise TypeError("grid_gdf must be a geopandas.GeoDataFrame")

    cfg = config or {}
    weights: Dict[str, Any] = cfg.get("feature_weights", {}) if isinstance(cfg, dict) else {}

    w_pop = float(weights.get("population", 0.35))
    w_poi = float(weights.get("poi", 0.25))
    w_traffic = float(weights.get("traffic", 0.25))
    w_charger_penalty = float(
        weights.get(
            "existing_chargers_penalty",
            weights.get("chargers_penalty", weights.get("charger_penalty", 0.15)),
        )
    )

    logger.info(
        "Computing demand score with weights -> population: %.2f, poi: %.2f, "
        "traffic: %.2f, charger_penalty: %.2f",
        w_pop,
        w_poi,
        w_traffic,
        w_charger_penalty,
    )

    gdf = grid_gdf.copy()

    # Retrieve or normalize feature columns
    pop_norm = _get_feature_series(gdf, "population_proxy_norm", "population_proxy")
    poi_count_norm = _get_feature_series(gdf, "poi_count_norm", "poi_count")
    poi_div_norm = _get_feature_series(gdf, "poi_diversity_norm", "poi_diversity")
    traffic_norm = _get_feature_series(gdf, "traffic_proxy_norm", "traffic_proxy")
    charger_norm = _get_feature_series(gdf, "chargers_in_radius_norm", "chargers_in_radius")

    # Combine POI metrics
    poi_combined = (poi_count_norm + poi_div_norm) / 2.0

    # Compute weighted demand score
    raw_demand = (
        w_pop * pop_norm
        + w_poi * poi_combined
        + w_traffic * traffic_norm
        - w_charger_penalty * charger_norm
    )

    gdf["demand_score"] = np.clip(raw_demand, 0.0, 1.0).astype(float)

    logger.info(
        "Demand score computed for %d cells (min=%.4f, mean=%.4f, max=%.4f)",
        len(gdf),
        gdf["demand_score"].min(),
        gdf["demand_score"].mean(),
        gdf["demand_score"].max(),
    )

    return gdf


def compute_supply_gap(
    grid_gdf: gpd.GeoDataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> gpd.GeoDataFrame:
    """Compute unmet EV charging demand (supply gap) per grid cell.

    Formula:
        unmet_demand = demand_score * (1 - chargers_in_radius_norm)

    This reduces demand for cells that are already well-served by existing chargers.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Spatial grid cells. If 'demand_score' column is missing, it is automatically
        computed using compute_demand_score.
    config : Optional[Dict[str, Any]]
        Configuration dictionary passed to compute_demand_score if needed.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of grid_gdf with added 'unmet_demand' column.
    """
    if not isinstance(grid_gdf, gpd.GeoDataFrame):
        raise TypeError("grid_gdf must be a geopandas.GeoDataFrame")

    gdf = grid_gdf.copy()

    # Ensure demand_score is available
    if "demand_score" not in gdf.columns:
        logger.info("'demand_score' not found; computing demand score first.")
        gdf = compute_demand_score(gdf, config=config)

    charger_norm = _get_feature_series(gdf, "chargers_in_radius_norm", "chargers_in_radius")

    # Compute unmet demand (supply gap)
    unmet_demand = gdf["demand_score"] * (1.0 - charger_norm)
    gdf["unmet_demand"] = np.clip(unmet_demand, 0.0, 1.0).astype(float)

    logger.info(
        "Supply gap computed for %d cells (min=%.4f, mean=%.4f, max=%.4f)",
        len(gdf),
        gdf["unmet_demand"].min(),
        gdf["unmet_demand"].mean(),
        gdf["unmet_demand"].max(),
    )

    return gdf


def save_demand_heatmap(
    grid_gdf: gpd.GeoDataFrame,
    base_dir: Union[Path, str],
) -> Path:
    """Create and save a matplotlib heatmap showing demand and supply gap.

    Saves the figure to outputs/demand_heatmap.png under base_dir.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Spatial grid cells containing 'demand_score' and optionally 'unmet_demand'.
    base_dir : Union[Path, str]
        Base directory of the project or outputs directory.

    Returns
    -------
    Path
        Absolute path to the saved heatmap image.
    """
    if not isinstance(grid_gdf, gpd.GeoDataFrame):
        raise TypeError("grid_gdf must be a geopandas.GeoDataFrame")

    base_path = Path(base_dir).resolve()
    out_dir = base_path if base_path.name == "outputs" else base_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "demand_heatmap.png"

    has_unmet = "unmet_demand" in grid_gdf.columns
    has_demand = "demand_score" in grid_gdf.columns

    if not has_demand and not has_unmet:
        logger.warning(
            "Neither 'demand_score' nor 'unmet_demand' found. Using default column."
        )

    # Determine plot layout
    if has_demand and has_unmet:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        ax_demand, ax_unmet = axes[0], axes[1]

        # Demand score plot
        grid_gdf.plot(
            column="demand_score",
            cmap="YlOrRd",
            linewidth=0.1,
            edgecolor="lightgrey",
            legend=True,
            legend_kwds={"label": "Demand Score [0 - 1]", "shrink": 0.65},
            ax=ax_demand,
        )
        ax_demand.set_title("EV Charging Demand Score", fontsize=14, fontweight="bold")
        ax_demand.set_axis_off()

        # Unmet demand plot
        grid_gdf.plot(
            column="unmet_demand",
            cmap="viridis",
            linewidth=0.1,
            edgecolor="lightgrey",
            legend=True,
            legend_kwds={"label": "Unmet Demand (Supply Gap) [0 - 1]", "shrink": 0.65},
            ax=ax_unmet,
        )
        ax_unmet.set_title("Unmet EV Demand (Supply Gap)", fontsize=14, fontweight="bold")
        ax_unmet.set_axis_off()

        fig.suptitle("Spatial EV Demand & Supply Gap Analysis", fontsize=16, fontweight="bold", y=0.98)
    else:
        target_col = "demand_score" if has_demand else ("unmet_demand" if has_unmet else grid_gdf.columns[0])
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        grid_gdf.plot(
            column=target_col,
            cmap="YlOrRd",
            linewidth=0.1,
            edgecolor="lightgrey",
            legend=True,
            legend_kwds={"label": f"{target_col} [0 - 1]", "shrink": 0.75},
            ax=ax,
        )
        ax.set_title(f"EV Charging Heatmap ({target_col})", fontsize=14, fontweight="bold")
        ax.set_axis_off()

    plt.tight_layout()
    try:
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        logger.info("Demand heatmap saved successfully to %s", out_file)
    except Exception as exc:
        logger.error("Failed to save demand heatmap to %s: %s", out_file, exc)
        raise
    finally:
        plt.close(fig)

    return out_file
