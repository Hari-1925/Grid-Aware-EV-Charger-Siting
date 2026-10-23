"""
Metrics Module for EV Charger Siting Optimization.

Computes pre- and post-optimization coverage, distance statistics,
population served, and grid proximity indicators, and exports results
to disk in standard formats.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _to_metric_crs(gdf: gpd.GeoDataFrame, target_crs: Optional[Any] = None) -> gpd.GeoDataFrame:
    """
    Ensure GeoDataFrame is in a projected metric CRS for accurate distance calculations.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Spatial dataframe to project.
    target_crs : Any, optional
        Target metric CRS. If None, estimates UTM or falls back to EPSG:3857.

    Returns
    -------
    gpd.GeoDataFrame
        Reprojected GeoDataFrame in a metric coordinate reference system.
    """
    if gdf is None or gdf.empty:
        return gdf

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    if gdf.crs.is_geographic:
        if target_crs is not None:
            return gdf.to_crs(target_crs)
        try:
            utm = gdf.estimate_utm_crs()
            return gdf.to_crs(utm)
        except Exception as e:
            logger.debug("estimate_utm_crs failed (%s), defaulting to EPSG:3857", e)
            return gdf.to_crs(epsg=3857)

    return gdf


def _extract_xy_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extract 2D coordinate array (x, y) from point or polygon geometry centroids.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Spatial dataframe with geometry column.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with Euclidean coordinates.
    """
    if gdf is None or gdf.empty:
        return np.empty((0, 2), dtype=float)

    # Use centroid for polygons/multipolygons; for points, centroid is the point itself
    centroids = gdf.geometry.centroid
    return np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])


def _clean_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy scalars and special floats to JSON-serializable types.

    Parameters
    ----------
    obj : Any
        Object or dictionary/list structure to sanitize.

    Returns
    -------
    Any
        JSON-compliant primitive types.
    """
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray, list)):
        return [_clean_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    return obj


def compute_metrics(
    grid_gdf: gpd.GeoDataFrame,
    recommended_sites_gdf: gpd.GeoDataFrame,
    chargers_gdf: Optional[gpd.GeoDataFrame],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute before-and-after spatial coverage and grid metrics.

    Calculates:
    - Before metrics (with existing chargers only):
      * % of demand cells covered (within coverage_radius_m of a charger)
      * Average distance to nearest charger
    - After metrics (with existing + recommended chargers):
      * % of demand cells covered
      * Average distance to nearest charger
    - Improvement:
      * Number of newly covered cells
      * Estimated newly covered population (sum of population_proxy in newly covered cells)
      * % of recommended sites within max_substation_dist_m

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Grid cells containing demand, geometry, and optional population proxy.
    recommended_sites_gdf : gpd.GeoDataFrame
        Optimal charger sites selected by the optimization model.
    chargers_gdf : Optional[gpd.GeoDataFrame]
        Existing EV charger locations.
    config : Dict[str, Any]
        Configuration dictionary containing 'coverage_radius_m',
        'max_substation_dist_m', etc.

    Returns
    -------
    Dict[str, Any]
        Structured dictionary containing before, after, and improvement metrics.
    """
    try:
        coverage_radius_m = float(config.get("coverage_radius_m", 1500.0))
        max_substation_dist_m = float(config.get("max_substation_dist_m", 3000.0))

        if grid_gdf is None or grid_gdf.empty:
            raise ValueError("grid_gdf cannot be empty when computing metrics.")

        # Project grid to metric CRS
        grid_proj = _to_metric_crs(grid_gdf)
        metric_crs = grid_proj.crs
        grid_coords = _extract_xy_coords(grid_proj)
        n_cells = len(grid_coords)

        # Identify population column
        pop_col = None
        for col in ["population_proxy", "population", "pop_proxy", "pop", "demand"]:
            if col in grid_gdf.columns:
                pop_col = col
                break

        # ---------------------------------------------------------
        # 1. Existing Chargers (Before)
        # ---------------------------------------------------------
        has_existing_chargers = chargers_gdf is not None and len(chargers_gdf) > 0
        if has_existing_chargers:
            chargers_proj = _to_metric_crs(chargers_gdf, target_crs=metric_crs)
            charger_coords = _extract_xy_coords(chargers_proj)
            charger_tree = cKDTree(charger_coords)

            dists_before, _ = charger_tree.query(grid_coords, k=1)
            covered_before_mask = dists_before <= coverage_radius_m
            pct_covered_before = float((np.sum(covered_before_mask) / n_cells) * 100.0)
            avg_dist_before = float(np.mean(dists_before))
        else:
            charger_coords = np.empty((0, 2), dtype=float)
            dists_before = np.full(n_cells, np.inf)
            covered_before_mask = np.zeros(n_cells, dtype=bool)
            pct_covered_before = 0.0
            avg_dist_before = float("inf")

        # ---------------------------------------------------------
        # 2. Recommended Sites + Existing (After)
        # ---------------------------------------------------------
        has_recommended = recommended_sites_gdf is not None and len(recommended_sites_gdf) > 0
        if has_recommended:
            rec_proj = _to_metric_crs(recommended_sites_gdf, target_crs=metric_crs)
            rec_coords = _extract_xy_coords(rec_proj)
        else:
            rec_coords = np.empty((0, 2), dtype=float)

        if len(rec_coords) > 0 or len(charger_coords) > 0:
            if len(charger_coords) > 0 and len(rec_coords) > 0:
                combined_coords = np.vstack([charger_coords, rec_coords])
            elif len(rec_coords) > 0:
                combined_coords = rec_coords
            else:
                combined_coords = charger_coords

            combined_tree = cKDTree(combined_coords)
            dists_after, _ = combined_tree.query(grid_coords, k=1)
            covered_after_mask = dists_after <= coverage_radius_m
            pct_covered_after = float((np.sum(covered_after_mask) / n_cells) * 100.0)
            avg_dist_after = float(np.mean(dists_after))
        else:
            dists_after = np.full(n_cells, np.inf)
            covered_after_mask = np.zeros(n_cells, dtype=bool)
            pct_covered_after = 0.0
            avg_dist_after = float("inf")

        # ---------------------------------------------------------
        # 3. Improvement Metrics
        # ---------------------------------------------------------
        newly_covered_mask = (~covered_before_mask) & covered_after_mask
        n_newly_covered = int(np.sum(newly_covered_mask))

        if pop_col is not None:
            newly_covered_population = float(grid_gdf.loc[newly_covered_mask, pop_col].sum())
        else:
            newly_covered_population = 0.0

        # Substation proximity check
        if has_recommended:
            sub_col = None
            for col in ["dist_to_substation", "dist_substation", "substation_dist_m"]:
                if col in recommended_sites_gdf.columns:
                    sub_col = col
                    break

            if sub_col is not None:
                dists_sub = recommended_sites_gdf[sub_col].to_numpy(dtype=float)
                within_sub = dists_sub <= max_substation_dist_m
                pct_within_substation = float(np.mean(within_sub) * 100.0)
            else:
                # If column not present, sites passed feasibility filter
                pct_within_substation = 100.0
        else:
            pct_within_substation = 0.0

        coverage_increase_pct = float(pct_covered_after - pct_covered_before)
        avg_distance_reduction_m = (
            float(avg_dist_before - avg_dist_after) if avg_dist_before != float("inf") else 0.0
        )

        metrics = {
            "before": {
                "pct_cells_covered": round(pct_covered_before, 2),
                "avg_distance_to_nearest_m": (
                    round(avg_dist_before, 1) if avg_dist_before != float("inf") else None
                ),
            },
            "after": {
                "pct_cells_covered": round(pct_covered_after, 2),
                "avg_distance_to_nearest_m": (
                    round(avg_dist_after, 1) if avg_dist_after != float("inf") else None
                ),
            },
            "improvement": {
                "newly_covered_cells": n_newly_covered,
                "newly_covered_population": round(newly_covered_population, 1),
                "pct_recommended_within_substation_dist": round(pct_within_substation, 1),
                "coverage_increase_pct": round(coverage_increase_pct, 2),
                "avg_distance_reduction_m": round(avg_distance_reduction_m, 1),
            },
            "total_grid_cells": n_cells,
            "coverage_radius_m": coverage_radius_m,
            "max_substation_dist_m": max_substation_dist_m,
            "n_recommended_sites": len(recommended_sites_gdf) if recommended_sites_gdf is not None else 0,
        }

        # Convenience top-level keys
        metrics["pct_cells_covered_before"] = metrics["before"]["pct_cells_covered"]
        metrics["avg_distance_nearest_before_m"] = metrics["before"]["avg_distance_to_nearest_m"]
        metrics["pct_cells_covered_after"] = metrics["after"]["pct_cells_covered"]
        metrics["avg_distance_nearest_after_m"] = metrics["after"]["avg_distance_to_nearest_m"]
        metrics["newly_covered_cells"] = metrics["improvement"]["newly_covered_cells"]
        metrics["newly_covered_population"] = metrics["improvement"]["newly_covered_population"]
        metrics["pct_recommended_within_substation_dist"] = metrics["improvement"]["pct_recommended_within_substation_dist"]

        logger.info(
            "Metrics calculated successfully: coverage before=%.2f%%, after=%.2f%%, newly covered cells=%d",
            pct_covered_before,
            pct_covered_after,
            n_newly_covered,
        )
        return metrics

    except Exception as e:
        logger.error("Failed to compute metrics: %s", str(e), exc_info=True)
        raise RuntimeError(f"Error computing siting metrics: {e}") from e


def save_metrics(
    metrics: Dict[str, Any],
    recommended_sites_gdf: gpd.GeoDataFrame,
    base_dir: Union[str, Path],
) -> None:
    """
    Save computed metrics to JSON and recommended sites to CSV.

    Exports:
    - outputs/metrics.json
    - outputs/recommended_sites.csv with columns:
      [rank, lat, lon, h3_index, demand_covered, dist_to_substation]

    Also logs key summary metrics to console.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of siting metrics from compute_metrics().
    recommended_sites_gdf : gpd.GeoDataFrame
        Selected optimal site locations.
    base_dir : Union[str, Path]
        Project root path.
    """
    try:
        base_path = Path(base_dir)
        output_dir = base_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "metrics.json"
        csv_path = output_dir / "recommended_sites.csv"

        # ---------------------------------------------------------
        # 1. Save Metrics JSON
        # ---------------------------------------------------------
        cleaned_metrics = _clean_for_json(metrics)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_metrics, f, indent=2)
        logger.info("Saved metrics to %s", json_path)

        # ---------------------------------------------------------
        # 2. Format & Save Recommended Sites CSV
        # ---------------------------------------------------------
        if recommended_sites_gdf is not None and not recommended_sites_gdf.empty:
            df = recommended_sites_gdf.copy()

            # Ensure WGS84 for lat/lon extraction
            if df.crs is not None and not df.crs.is_geographic:
                df_geo = df.to_crs(epsg=4326)
            elif df.crs is None:
                df_geo = df.set_crs(epsg=4326)
            else:
                df_geo = df

            # Determine coordinates
            centroids = df_geo.geometry.centroid
            lats = centroids.y.to_numpy()
            lons = centroids.x.to_numpy()

            n_sites = len(df)

            # 1. rank
            if "rank" in df.columns:
                ranks = df["rank"].to_numpy()
            else:
                ranks = np.arange(1, n_sites + 1)

            # 2. h3_index
            if "h3_index" in df.columns:
                h3_indices = df["h3_index"].astype(str).to_numpy()
            elif df.index.name == "h3_index":
                h3_indices = df.index.astype(str).to_numpy()
            elif isinstance(df.index[0], str) and len(df.index[0]) == 15:
                h3_indices = df.index.astype(str).to_numpy()
            else:
                h3_indices = np.array([f"site_{i}" for i in range(1, n_sites + 1)])

            # 3. demand_covered
            demand_col = None
            for col in ["demand_covered", "covered_demand", "unmet_demand", "demand", "score"]:
                if col in df.columns:
                    demand_col = col
                    break
            if demand_col is not None:
                demand_covered = np.round(df[demand_col].to_numpy(dtype=float), 4)
            else:
                demand_covered = np.zeros(n_sites, dtype=float)

            # 4. dist_to_substation
            sub_col = None
            for col in ["dist_to_substation", "dist_substation", "substation_dist_m"]:
                if col in df.columns:
                    sub_col = col
                    break
            if sub_col is not None:
                dist_substation = np.round(df[sub_col].to_numpy(dtype=float), 2)
            else:
                dist_substation = np.zeros(n_sites, dtype=float)

            export_df = pd.DataFrame(
                {
                    "rank": ranks,
                    "lat": np.round(lats, 6),
                    "lon": np.round(lons, 6),
                    "h3_index": h3_indices,
                    "demand_covered": demand_covered,
                    "dist_to_substation": dist_substation,
                }
            )

            # Sort by rank
            export_df.sort_values(by="rank", inplace=True)
            export_df.to_csv(csv_path, index=False)
            logger.info("Saved %d recommended sites to %s", len(export_df), csv_path)
        else:
            # Create empty CSV with expected columns
            empty_df = pd.DataFrame(
                columns=["rank", "lat", "lon", "h3_index", "demand_covered", "dist_to_substation"]
            )
            empty_df.to_csv(csv_path, index=False)
            logger.warning("Recommended sites GeoDataFrame was empty; saved headers to %s", csv_path)

        # ---------------------------------------------------------
        # 3. Log Key Metrics to Console
        # ---------------------------------------------------------
        pct_before = metrics.get("pct_cells_covered_before", metrics.get("before", {}).get("pct_cells_covered", 0.0))
        pct_after = metrics.get("pct_cells_covered_after", metrics.get("after", {}).get("pct_cells_covered", 0.0))
        new_cells = metrics.get("newly_covered_cells", metrics.get("improvement", {}).get("newly_covered_cells", 0))
        new_pop = metrics.get("newly_covered_population", metrics.get("improvement", {}).get("newly_covered_population", 0.0))
        pct_sub = metrics.get(
            "pct_recommended_within_substation_dist",
            metrics.get("improvement", {}).get("pct_recommended_within_substation_dist", 100.0),
        )

        logger.info("==================================================")
        logger.info("       OPTIMAL EV CHARGER SITING METRICS         ")
        logger.info("==================================================")
        logger.info("Coverage Before Optimization : %6.2f %%", pct_before)
        logger.info("Coverage After Optimization  : %6.2f %%", pct_after)
        logger.info("Coverage Improvement         : +%5.2f %%", pct_after - pct_before)
        logger.info("Newly Covered Grid Cells     : %d", new_cells)
        logger.info("Estimated Population Served  : %s", f"{new_pop:,.0f}")
        logger.info("Substation Distance Feasible : %6.1f %%", pct_sub)
        logger.info("Results Saved: %s", str(json_path))
        logger.info("==================================================")

    except Exception as e:
        logger.error("Failed to save metrics or recommended sites: %s", str(e), exc_info=True)
        raise IOError(f"Error saving siting outputs: {e}") from e
