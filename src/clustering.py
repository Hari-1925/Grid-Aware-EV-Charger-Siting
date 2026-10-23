"""Spatial clustering of high-demand cells to identify EV charger candidate sites.

This module clusters high unmet demand cells using K-Means and HDBSCAN,
snaps cluster centroids to nearest spatial grid cells via KDTree, and aggregates
candidate sites for subsequent grid-aware optimization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

# Scoring module import handling
try:
    from src.scoring import compute_supply_gap
except ImportError:
    try:
        from scoring import compute_supply_gap
    except ImportError:
        compute_supply_gap = None

logger = logging.getLogger(__name__)


def _extract_h3_series(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Extract or generate H3 index strings for GeoDataFrame rows.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing cell attributes or index.

    Returns
    -------
    pd.Series
        Series of H3 index strings.
    """
    if "h3_index" in gdf.columns:
        return gdf["h3_index"].astype(str)
    if "hex_id" in gdf.columns:
        return gdf["hex_id"].astype(str)
    if gdf.index.name in ["h3_index", "hex_id"]:
        return gdf.index.astype(str)

    if len(gdf) > 0:
        first_val = str(gdf.index[0])
        if len(first_val) == 15 and all(c in "0123456789abcdefABCDEF" for c in first_val):
            return gdf.index.astype(str)

    return pd.Series([f"cell_{i}" for i in range(len(gdf))], index=gdf.index, dtype=str)


def _prepare_utm_projection(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Any]:
    """Ensure GeoDataFrame has a valid CRS and project to UTM meters.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Spatial GeoDataFrame.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, Any]
        Projected GeoDataFrame in UTM and the UTM CRS object.
    """
    working_gdf = gdf.copy()
    if working_gdf.crs is None:
        logger.warning("CRS not defined on GeoDataFrame. Assuming EPSG:4326.")
        working_gdf.set_crs(epsg=4326, inplace=True)

    if working_gdf.crs.is_geographic:
        try:
            utm_crs = working_gdf.estimate_utm_crs()
            logger.info("Projecting geographic coordinates to UTM CRS: %s", utm_crs)
            return working_gdf.to_crs(utm_crs), utm_crs
        except Exception as exc:
            logger.warning(
                "Could not estimate UTM CRS (%s). Falling back to EPSG:3857.", exc
            )
            return working_gdf.to_crs(epsg=3857), "EPSG:3857"

    return working_gdf, working_gdf.crs


def snap_centroids_to_cells(
    cluster_centers_utm: np.ndarray,
    cluster_labels: List[int],
    grid_gdf: gpd.GeoDataFrame,
    grid_gdf_utm: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Snap Euclidean cluster centers to nearest grid cell centroids via cKDTree.

    Parameters
    ----------
    cluster_centers_utm : np.ndarray
        Array of (N, 2) UTM coordinates of cluster centers.
    cluster_labels : List[int]
        Corresponding cluster label IDs.
    grid_gdf : gpd.GeoDataFrame
        Original grid cells in source CRS.
    grid_gdf_utm : gpd.GeoDataFrame
        Grid cells projected to UTM.

    Returns
    -------
    gpd.GeoDataFrame
        Snapped cluster centroids with columns: site_id, geometry, h3_index,
        unmet_demand, cluster_label.
    """
    if len(cluster_centers_utm) == 0:
        return gpd.GeoDataFrame(
            columns=["site_id", "geometry", "h3_index", "unmet_demand", "cluster_label"],
            geometry="geometry",
            crs=grid_gdf.crs,
        )

    # Extract all cell centroids in UTM for KDTree indexing
    grid_centroids_utm = grid_gdf_utm.geometry.centroid
    all_utm_coords = np.column_stack([
        grid_centroids_utm.x.to_numpy(),
        grid_centroids_utm.y.to_numpy(),
    ])

    kdtree = cKDTree(all_utm_coords)
    _, nearest_indices = kdtree.query(cluster_centers_utm)

    # Query nearest cells in original CRS
    snapped_cells = grid_gdf.iloc[nearest_indices].copy()
    h3_indices = _extract_h3_series(snapped_cells).values

    snapped_centroids = gpd.GeoDataFrame(
        {
            "site_id": [f"cluster_{lbl}" for lbl in cluster_labels],
            "geometry": snapped_cells.geometry.centroid.values,
            "h3_index": h3_indices,
            "unmet_demand": snapped_cells["unmet_demand"].to_numpy(dtype=float),
            "cluster_label": cluster_labels,
        },
        geometry="geometry",
        crs=grid_gdf.crs,
    )

    logger.info("Successfully snapped %d cluster centroids to nearest grid cells.", len(snapped_centroids))
    return snapped_centroids


def cluster_demand_cells(
    grid_gdf: gpd.GeoDataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> gpd.GeoDataFrame:
    """Cluster top unmet-demand cells to generate candidate EV charger sites.

    Process:
    1. Select top `top_demand_fraction` (default 30%) cells by unmet_demand.
    2. Extract UTM coordinates and unmet_demand as sample_weight.
    3. Run K-Means with n_clusters = int(n_new_stations * kmeans_n_clusters_ratio).
    4. Run HDBSCAN with min_cluster_size and min_samples from config.
    5. Compare: if HDBSCAN produces >= n_new_stations meaningful clusters (not noise),
       use it; else fall back to K-Means.
    6. Add 'cluster_label' column to the top cells.
    7. Snap cluster centroids to nearest cell centroid via scipy.spatial.cKDTree.
    8. Combine snapped centroids with all high-demand cell centroids as candidates.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Spatial grid cells containing demand metrics.
    config : Optional[Dict[str, Any]]
        Configuration parameters for clustering and station targets.

    Returns
    -------
    gpd.GeoDataFrame
        Candidate sites GeoDataFrame with columns: site_id, geometry, h3_index,
        unmet_demand.
    """
    if not isinstance(grid_gdf, gpd.GeoDataFrame):
        raise TypeError("grid_gdf must be a geopandas.GeoDataFrame")

    cfg = config or {}
    clustering_cfg: Dict[str, Any] = (
        cfg.get("clustering", {}) if isinstance(cfg, dict) else {}
    )

    top_demand_fraction = float(clustering_cfg.get("top_demand_fraction", 0.30))
    n_new_stations = int(cfg.get("n_new_stations", 20) if isinstance(cfg, dict) else 20)
    kmeans_ratio = float(clustering_cfg.get("kmeans_n_clusters_ratio", 1.5))
    hdbscan_min_cluster_size = int(clustering_cfg.get("hdbscan_min_cluster_size", 5))
    hdbscan_min_samples = int(clustering_cfg.get("hdbscan_min_samples", 3))

    working_gdf = grid_gdf.copy()

    # Ensure unmet_demand is available
    if "unmet_demand" not in working_gdf.columns:
        if compute_supply_gap is not None:
            logger.info("'unmet_demand' column missing; computing supply gap.")
            working_gdf = compute_supply_gap(working_gdf, config=cfg)
        elif "demand_score" in working_gdf.columns:
            logger.warning("compute_supply_gap unavailable; using 'demand_score' as unmet_demand.")
            working_gdf["unmet_demand"] = working_gdf["demand_score"]
        else:
            raise KeyError("grid_gdf must contain 'unmet_demand' or 'demand_score' column.")

    # 1. Select top demand fraction
    n_top = max(1, int(np.ceil(len(working_gdf) * top_demand_fraction)))
    top_cells = working_gdf.nlargest(n_top, "unmet_demand").copy()
    logger.info(
        "Selected top %d cells (%.1f%%) for clustering. Min unmet_demand in top cells: %.4f",
        len(top_cells),
        top_demand_fraction * 100,
        top_cells["unmet_demand"].min(),
    )

    # 2. Project to UTM and extract coordinates and sample weights
    working_gdf_utm, utm_crs = _prepare_utm_projection(working_gdf)
    top_cells_utm = top_cells.to_crs(utm_crs)

    top_centroids_utm = top_cells_utm.geometry.centroid
    coords_utm = np.column_stack([
        top_centroids_utm.x.to_numpy(),
        top_centroids_utm.y.to_numpy(),
    ])

    sample_weights = top_cells["unmet_demand"].to_numpy(dtype=float)
    sample_weights = np.nan_to_num(sample_weights, nan=0.0)
    sample_weights = np.clip(sample_weights, a_min=1e-6, a_max=None)

    # 3. Run K-Means
    n_kmeans_clusters = max(1, min(int(np.ceil(n_new_stations * kmeans_ratio)), len(top_cells)))
    kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(coords_utm, sample_weight=sample_weights)

    # 4. Run HDBSCAN
    hdb_labels: Optional[np.ndarray] = None
    min_cluster_size = max(2, min(hdbscan_min_cluster_size, len(top_cells)))
    min_samples = max(1, min(hdbscan_min_samples, len(top_cells)))

    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        hdb_labels = clusterer.fit_predict(coords_utm)
        logger.debug("HDBSCAN from hdbscan library completed.")
    except Exception as exc1:
        logger.debug("Standard hdbscan import failed: %s. Attempting sklearn.cluster.HDBSCAN.", exc1)
        try:
            from sklearn.cluster import HDBSCAN  # type: ignore

            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
            )
            hdb_labels = clusterer.fit_predict(coords_utm)
            logger.debug("HDBSCAN from scikit-learn completed.")
        except Exception as exc2:
            logger.warning("HDBSCAN clustering failed: %s. Proceeding with K-Means.", exc2)
            hdb_labels = None

    # 5. Compare HDBSCAN vs K-Means
    use_hdbscan = False
    if hdb_labels is not None:
        meaningful_clusters = [int(lbl) for lbl in np.unique(hdb_labels) if lbl >= 0]
        n_meaningful = len(meaningful_clusters)
        logger.info(
            "HDBSCAN produced %d meaningful clusters (threshold: >= %d stations).",
            n_meaningful,
            n_new_stations,
        )
        if n_meaningful >= n_new_stations:
            use_hdbscan = True
            chosen_labels = hdb_labels
            active_clusters = meaningful_clusters
            chosen_method = "HDBSCAN"
        else:
            logger.info("HDBSCAN cluster count < n_new_stations. Using K-Means.")

    if not use_hdbscan:
        chosen_labels = kmeans_labels
        active_clusters = [int(lbl) for lbl in np.unique(kmeans_labels)]
        chosen_method = "KMeans"

    logger.info("Clustering algorithm selected: %s (%d clusters).", chosen_method, len(active_clusters))

    # 6. Add 'cluster_label' column to the top cells
    top_cells["cluster_label"] = chosen_labels

    # 7. Compute weighted cluster centers in UTM
    cluster_centers: List[List[float]] = []
    cluster_labels_list: List[int] = []

    for lbl in active_clusters:
        mask = (chosen_labels == lbl)
        pts = coords_utm[mask]
        w = sample_weights[mask]
        w_sum = w.sum()
        if w_sum > 0:
            cx = float((pts[:, 0] * w).sum() / w_sum)
            cy = float((pts[:, 1] * w).sum() / w_sum)
        else:
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
        cluster_centers.append([cx, cy])
        cluster_labels_list.append(lbl)

    cluster_centers_arr = np.array(cluster_centers)

    # Snap cluster centroids to nearest cell centroid via cKDTree
    snapped_centroids = snap_centroids_to_cells(
        cluster_centers_arr,
        cluster_labels_list,
        working_gdf,
        working_gdf_utm,
    )

    # 8. Combine cluster centroids + high-demand cell centroids as candidate sites
    candidate_sites = get_candidate_sites(working_gdf, snapped_centroids, config=cfg)
    return candidate_sites


def get_candidate_sites(
    grid_gdf: gpd.GeoDataFrame,
    cluster_centroids: gpd.GeoDataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> gpd.GeoDataFrame:
    """Combine cluster centroids and top unmet_demand cell centroids into candidate sites.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Spatial grid cells containing 'unmet_demand'.
    cluster_centroids : gpd.GeoDataFrame
        GeoDataFrame of cluster centroids snapped to grid cells.
    config : Optional[Dict[str, Any]]
        Configuration containing 'clustering.top_demand_fraction'.

    Returns
    -------
    gpd.GeoDataFrame
        Deduplicated candidate sites with columns:
        site_id, geometry, h3_index, unmet_demand.
    """
    if not isinstance(grid_gdf, gpd.GeoDataFrame):
        raise TypeError("grid_gdf must be a geopandas.GeoDataFrame")

    cfg = config or {}
    clustering_cfg: Dict[str, Any] = (
        cfg.get("clustering", {}) if isinstance(cfg, dict) else {}
    )
    top_fraction = float(clustering_cfg.get("top_demand_fraction", 0.30))

    working_gdf = grid_gdf.copy()
    if "unmet_demand" not in working_gdf.columns:
        if compute_supply_gap is not None:
            working_gdf = compute_supply_gap(working_gdf, config=cfg)
        elif "demand_score" in working_gdf.columns:
            working_gdf["unmet_demand"] = working_gdf["demand_score"]
        else:
            working_gdf["unmet_demand"] = 0.0

    # Extract top unmet-demand cells
    n_top = max(1, int(np.ceil(len(working_gdf) * top_fraction)))
    top_cells = working_gdf.nlargest(n_top, "unmet_demand").copy()

    # Build candidates from top demand cells
    top_candidates = gpd.GeoDataFrame(
        {
            "site_id": [f"demand_{i}" for i in range(len(top_cells))],
            "geometry": top_cells.geometry.centroid.values,
            "h3_index": _extract_h3_series(top_cells).values,
            "unmet_demand": top_cells["unmet_demand"].to_numpy(dtype=float),
        },
        geometry="geometry",
        crs=grid_gdf.crs,
    )

    # Process cluster centroids
    if cluster_centroids is not None and len(cluster_centroids) > 0:
        centroids_h3 = _extract_h3_series(cluster_centroids).values
        centroids_unmet = (
            cluster_centroids["unmet_demand"].to_numpy(dtype=float)
            if "unmet_demand" in cluster_centroids.columns
            else np.zeros(len(cluster_centroids), dtype=float)
        )
        centroids_geom = cluster_centroids.geometry.centroid.values

        cluster_candidates = gpd.GeoDataFrame(
            {
                "site_id": [f"cluster_{i}" for i in range(len(cluster_centroids))],
                "geometry": centroids_geom,
                "h3_index": centroids_h3,
                "unmet_demand": centroids_unmet,
            },
            geometry="geometry",
            crs=grid_gdf.crs,
        )
        # Cluster centroids first so they are prioritized in deduplication
        combined = pd.concat([cluster_candidates, top_candidates], ignore_index=True)
    else:
        combined = top_candidates

    # Remove duplicates by h3_index and coordinate proximity
    combined = combined.drop_duplicates(subset=["h3_index"], keep="first").copy()

    # Coordinate-based deduplication (rounded to ~1cm in degrees/meters)
    coord_keys = combined.geometry.apply(lambda p: (round(p.x, 6), round(p.y, 6)))
    combined = combined.drop_duplicates(subset=[coord_keys], keep="first").copy()

    # Assign clean, standardized sequential site_ids
    combined["site_id"] = [f"site_{i:04d}" for i in range(len(combined))]

    result_gdf = gpd.GeoDataFrame(
        combined[["site_id", "geometry", "h3_index", "unmet_demand"]],
        geometry="geometry",
        crs=grid_gdf.crs,
    )

    logger.info(
        "Candidate site generation complete: %d unique candidate sites identified.",
        len(result_gdf),
    )
    return result_gdf
