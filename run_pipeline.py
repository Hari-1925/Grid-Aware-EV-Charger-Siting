"""
Optimal EV Charger Placement Pipeline Runner.

This module orchestrates the complete end-to-end data-driven and grid-aware
electric vehicle (EV) charging station siting optimization pipeline:
    1. Ingestion: Download & cache OSM road networks, POIs, chargers, substations.
    2. Preprocessing: Clean, deduplicate, reproject, and buffer geospatial layers.
    3. Grid & Features: Discretize study area into H3 hexagonal cells and aggregate features.
    4. Scoring: Compute multi-criteria demand scores and unmet supply gaps.
    5. Clustering: Detect spatial demand clusters and extract candidate site coordinates.
    6. Feasibility Filtering: Enforce electrical substation proximity and spatial spacing.
    7. Optimization: Solve Maximal Covering Location Problem (MCLP) using PuLP.
    8. Validation: Conduct holdout validation, baseline comparisons, and sensitivity analysis.
    9. Metrics: Compute spatial coverage and network accessibility improvements.
    10. Map Visualization: Generate interactive Folium map artifact with togglable layers.

Author: EV Siting Optimization Team
"""

import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import yaml

# Lazy/resilient imports from the src package to allow pipeline orchestration
# even when components are tested or imported in varied environments.
try:
    from src.ingest import ingest_data
except ImportError:
    ingest_data = None  # type: ignore[assignment]

try:
    from src.preprocess import preprocess_data
except ImportError:
    preprocess_data = None  # type: ignore[assignment]

try:
    from src.grid_features import compute_cell_features, create_h3_grid
except ImportError:
    create_h3_grid = None  # type: ignore[assignment]
    compute_cell_features = None  # type: ignore[assignment]

try:
    from src.scoring import compute_demand_score, compute_supply_gap, save_demand_heatmap
except ImportError:
    compute_demand_score = None  # type: ignore[assignment]
    compute_supply_gap = None  # type: ignore[assignment]
    save_demand_heatmap = None  # type: ignore[assignment]

try:
    from src.clustering import cluster_demand_cells, get_candidate_sites
except ImportError:
    cluster_demand_cells = None  # type: ignore[assignment]
    get_candidate_sites = None  # type: ignore[assignment]

try:
    from src.optimize import feasibility_filter, solve_mclp
except ImportError:
    feasibility_filter = None  # type: ignore[assignment]
    solve_mclp = None  # type: ignore[assignment]

try:
    from src.validate import (
        baseline_comparison,
        holdout_validation,
        sensitivity_analysis,
    )
except ImportError:
    holdout_validation = None  # type: ignore[assignment]
    baseline_comparison = None  # type: ignore[assignment]
    sensitivity_analysis = None  # type: ignore[assignment]

try:
    from src.metrics import compute_metrics, save_metrics
except ImportError:
    compute_metrics = None  # type: ignore[assignment]
    save_metrics = None  # type: ignore[assignment]

logger = logging.getLogger("ev_siting_pipeline")


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure timestamped logging for console and file output.

    Args:
        log_file: Optional path to output log file.
        level: Logging level threshold (default: logging.INFO).

    Returns:
        logging.Logger: Configured pipeline logger.
    """
    pipe_logger = logging.getLogger("ev_siting_pipeline")
    pipe_logger.setLevel(level)

    # Avoid duplicate handlers if setup_logging is invoked repeatedly
    if not pipe_logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        pipe_logger.addHandler(stream_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            pipe_logger.addHandler(file_handler)

    return pipe_logger


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and validate pipeline configuration from YAML file.

    Args:
        config_path: Path to configuration YAML. Defaults to 'config.yaml'
            in the project root directory.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If YAML syntax parsing fails.
    """
    base_dir = Path(__file__).resolve().parent
    target_path = Path(config_path) if config_path else (base_dir / "config.yaml")

    if not target_path.exists():
        err_msg = f"Configuration file not found at: {target_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f) or {}
        logger.info("Loaded pipeline configuration from %s", target_path)
        return config
    except yaml.YAMLError as e:
        logger.error("Failed to parse YAML configuration: %s", e)
        raise


def create_directories(base_dir: Path) -> Dict[str, Path]:
    """Create all required input, cache, and output directory trees.

    Args:
        base_dir: Project root directory path.

    Returns:
        Dict[str, Path]: Mapping of directory identifiers to Path instances.
    """
    dirs = {
        "outputs": base_dir / "outputs",
        "data": base_dir / "data",
        "data_raw": base_dir / "data" / "raw",
        "data_processed": base_dir / "data" / "processed",
        "cache": base_dir / "data" / "cache",
    }
    for dir_name, path in dirs.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error("Failed creating directory %s (%s): %s", dir_name, path, e)
            raise
    return dirs


def _invoke_step(
    func: Optional[Callable[..., Any]],
    *preferred_args: Any,
    func_name: str = "step",
    **preferred_kwargs: Any,
) -> Any:
    """Invoke a modular pipeline step with flexible parameter binding.

    This helper inspects function signatures and dynamically matches supplied
    positional or keyword arguments. This prevents crashes due to parameter
    naming mismatches across modular components.

    Args:
        func: The callable pipeline step.
        *preferred_args: Ordered candidate positional arguments.
        func_name: Descriptive name for logging errors.
        **preferred_kwargs: Named candidate keyword arguments.

    Returns:
        Any: Result from the invoked callable.

    Raises:
        RuntimeError: If the function is not defined or callable.
        Exception: Re-raises execution errors with enriched context.
    """
    if func is None or not callable(func):
        raise RuntimeError(
            f"Pipeline function for '{func_name}' is not imported or callable."
        )

    try:
        # First attempt standard call with preferred arguments
        return func(*preferred_args, **preferred_kwargs)
    except TypeError as type_err:
        logger.debug(
            "Direct invocation of %s failed with TypeError: %s. Attempting signature adaptation.",
            func_name,
            type_err,
        )
        sig = inspect.signature(func)
        params = sig.parameters

        bound_args = []
        bound_kwargs = {}

        # 1. Match keyword arguments matching signature parameter names
        for p_name, param in params.items():
            if p_name in preferred_kwargs:
                bound_kwargs[p_name] = preferred_kwargs[p_name]

        # 2. Fill unassigned positional-or-keyword parameters from preferred_args
        arg_iter = iter(preferred_args)
        for p_name, param in params.items():
            if p_name not in bound_kwargs:
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    try:
                        bound_args.append(next(arg_iter))
                    except StopIteration:
                        break

        try:
            return func(*bound_args, **bound_kwargs)
        except Exception as retry_err:
            logger.error(
                "Invocation of %s failed after signature adaptation: %s",
                func_name,
                retry_err,
            )
            raise


def _extract_layer(
    data: Optional[Dict[str, Any]],
    *candidate_keys: str,
) -> Optional[gpd.GeoDataFrame]:
    """Extract a GeoDataFrame layer from a container dictionary.

    Args:
        data: Container dictionary containing geospatial datasets.
        *candidate_keys: Candidate key names in order of preference.

    Returns:
        Optional[gpd.GeoDataFrame]: Matched GeoDataFrame, or None if not found.
    """
    if not isinstance(data, dict):
        return None
    for key in candidate_keys:
        val = data.get(key)
        if isinstance(val, gpd.GeoDataFrame) and not val.empty:
            return val
        if isinstance(val, pd.DataFrame) and not val.empty and hasattr(val, "geometry"):
            return gpd.GeoDataFrame(val)
    return None


def _ensure_wgs84(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    """Ensure a GeoDataFrame uses EPSG:4326 (WGS84) geographic coordinates.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        Optional[gpd.GeoDataFrame]: Reprojected GeoDataFrame or None.
    """
    if gdf is None or gdf.empty or not hasattr(gdf, "geometry"):
        return gdf
    try:
        if gdf.crs is None:
            logger.warning("Layer missing CRS. Defaulting to EPSG:4326.")
            return gdf.set_crs(epsg=4326)
        if gdf.crs.to_epsg() != 4326:
            return gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        logger.warning("CRS reprojection to EPSG:4326 failed: %s", e)
        return gdf


def _compute_point_distances(
    points_a: gpd.GeoDataFrame,
    points_b: gpd.GeoDataFrame,
) -> np.ndarray:
    """Compute vectorized pairwise Euclidean distances in meters.

    Projects points to EPSG:3857 (Web Mercator) or planar coordinates to
    compute fast metric distances via scipy.spatial.distance.cdist.

    Args:
        points_a: First point GeoDataFrame (e.g. demand centroids).
        points_b: Second point GeoDataFrame (e.g. charging stations).

    Returns:
        np.ndarray: Pairwise distance matrix of shape (len(points_a), len(points_b)).
    """
    if points_a.empty or points_b.empty:
        return np.empty((len(points_a), len(points_b)))

    # Project to metric CRS for accurate distance evaluation
    a_proj = points_a.to_crs(epsg=3857) if points_a.crs and points_a.crs.to_epsg() != 3857 else points_a
    b_proj = points_b.to_crs(epsg=3857) if points_b.crs and points_b.crs.to_epsg() != 3857 else points_b

    coords_a = np.column_stack([a_proj.geometry.x.to_numpy(), a_proj.geometry.y.to_numpy()])
    coords_b = np.column_stack([b_proj.geometry.x.to_numpy(), b_proj.geometry.y.to_numpy()])

    return cdist(coords_a, coords_b)


def save_map(
    grid_gdf: Optional[gpd.GeoDataFrame] = None,
    selected_sites: Optional[gpd.GeoDataFrame] = None,
    candidate_sites: Optional[gpd.GeoDataFrame] = None,
    existing_chargers: Optional[gpd.GeoDataFrame] = None,
    substations: Optional[gpd.GeoDataFrame] = None,
    boundary: Optional[gpd.GeoDataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    output_path: Union[str, Path] = "outputs/map.html",
) -> folium.Map:
    """Create a rich, standalone interactive Folium map artifact.

    Builds an interactive multi-layer map containing:
      - City boundary polygon
      - Choropleth / styled H3 hexagon demand and supply-gap grid
      - Existing charging stations with details
      - Power substations (grid interconnection nodes)
      - Filtered candidate site locations
      - Selected optimal charger sites with rank and coverage metadata
      - 1500m coverage buffer radii around recommended stations
      - Interactive layer control and legend overlay

    Args:
        grid_gdf: Scored H3 hexagon grid GeoDataFrame.
        selected_sites: MCLP-selected optimal EV charger locations.
        candidate_sites: Feasible candidate site points.
        existing_chargers: Pre-existing EV charger points.
        substations: Power grid substations.
        boundary: Study area polygon boundary.
        config: Configuration dictionary.
        output_path: Path to save the resulting standalone HTML file.

    Returns:
        folium.Map: Generated Folium Map object.
    """
    logger.info("Assembling interactive Folium map artifact...")
    cfg = config or {}
    coverage_radius_m = float(cfg.get("coverage_radius_m", 1500.0))

    # Reproject all layers to EPSG:4326 for Folium compatibility
    grid_wgs = _ensure_wgs84(grid_gdf)
    sites_wgs = _ensure_wgs84(selected_sites)
    candidates_wgs = _ensure_wgs84(candidate_sites)
    chargers_wgs = _ensure_wgs84(existing_chargers)
    subs_wgs = _ensure_wgs84(substations)
    boundary_wgs = _ensure_wgs84(boundary)

    # Determine map center coordinate
    center_lat, center_lon = 30.2672, -97.7431  # Default Austin, TX
    if boundary_wgs is not None and not boundary_wgs.empty:
        centroid = boundary_wgs.geometry.unary_union.centroid
        center_lat, center_lon = centroid.y, centroid.x
    elif sites_wgs is not None and not sites_wgs.empty:
        center_lat = float(sites_wgs.geometry.y.mean())
        center_lon = float(sites_wgs.geometry.x.mean())
    elif chargers_wgs is not None and not chargers_wgs.empty:
        center_lat = float(chargers_wgs.geometry.y.mean())
        center_lon = float(chargers_wgs.geometry.x.mean())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Base Tile Options
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer("CartoDB dark_matter", name="CartoDB Dark Matter").add_to(fmap)

    # 1. Study Area Boundary Layer
    if boundary_wgs is not None and not boundary_wgs.empty:
        boundary_group = folium.FeatureGroup(name="Study Area Boundary", show=True)
        folium.GeoJson(
            boundary_wgs,
            style_function=lambda x: {
                "fillColor": "#34495e",
                "fillOpacity": 0.05,
                "color": "#2c3e50",
                "weight": 2.5,
                "dashArray": "6, 6",
            },
            tooltip="Study Area Boundary",
        ).add_to(boundary_group)
        boundary_group.add_to(fmap)

    # 2. Demand & Supply Gap H3 Hexagonal Grid
    if grid_wgs is not None and not grid_wgs.empty:
        grid_group = folium.FeatureGroup(name="Demand & Supply Gap Grid", show=True)

        score_col = "supply_gap" if "supply_gap" in grid_wgs.columns else "demand_score"
        if score_col in grid_wgs.columns:
            scores = grid_wgs[score_col].fillna(0.0).to_numpy()
            s_min, s_max = float(scores.min()), float(scores.max())
            s_range = s_max - s_min if s_max > s_min else 1.0

            def style_hex(feature: Dict[str, Any]) -> Dict[str, Any]:
                val = feature["properties"].get(score_col, 0.0) or 0.0
                norm = max(0.0, min(1.0, (val - s_min) / s_range))
                # Yellow to Red gradient interpolation
                r = int(255)
                g = int(255 * (1.0 - norm * 0.8))
                b = int(50 * (1.0 - norm))
                return {
                    "fillColor": f"#{r:02x}{g:02x}{b:02x}",
                    "fillOpacity": 0.45 * norm + 0.1,
                    "color": "#7f8c8d",
                    "weight": 0.5,
                }

            # Select tooltip fields
            tooltip_fields = [
                c for c in ["h3_index", "demand_score", "supply_gap", "population", "poi_count"]
                if c in grid_wgs.columns
            ]
            folium.GeoJson(
                grid_wgs,
                style_function=style_hex,
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=[f.replace("_", " ").title() + ":" for f in tooltip_fields],
                    localize=True,
                ) if tooltip_fields else None,
            ).add_to(grid_group)
            grid_group.add_to(fmap)

    # 3. Power Substations Layer
    if subs_wgs is not None and not subs_wgs.empty:
        subs_group = folium.FeatureGroup(name="Power Substations (Grid)", show=True)
        for _, row in subs_wgs.iterrows():
            geom = row.geometry
            if geom and hasattr(geom, "y") and hasattr(geom, "x"):
                sub_name = row.get("name", "Power Substation")
                voltage = row.get("voltage", "Standard MV")
                popup_html = (
                    f"<div style='font-family: Arial; font-size: 12px;'>"
                    f"<b style='color: #c0392b;'>⚡ Power Substation</b><br/>"
                    f"<b>Name:</b> {sub_name}<br/>"
                    f"<b>Voltage:</b> {voltage}<br/>"
                    f"<b>Coordinates:</b> {geom.y:.5f}, {geom.x:.5f}"
                    f"</div>"
                )
                folium.CircleMarker(
                    location=[geom.y, geom.x],
                    radius=6,
                    color="#962d22",
                    weight=1.5,
                    fill=True,
                    fill_color="#e74c3c",
                    fill_opacity=0.9,
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"Substation: {sub_name}",
                ).add_to(subs_group)
        subs_group.add_to(fmap)

    # 4. Existing EV Chargers Layer
    if chargers_wgs is not None and not chargers_wgs.empty:
        chargers_group = folium.FeatureGroup(name="Existing EV Chargers", show=True)
        for _, row in chargers_wgs.iterrows():
            geom = row.geometry
            if geom and hasattr(geom, "y") and hasattr(geom, "x"):
                c_name = row.get("name", row.get("operator", "Existing Station"))
                c_power = row.get("power", row.get("capacity", "Level 2 / DCFC"))
                popup_html = (
                    f"<div style='font-family: Arial; font-size: 12px;'>"
                    f"<b style='color: #2980b9;'>🔌 Existing EV Charger</b><br/>"
                    f"<b>Name:</b> {c_name}<br/>"
                    f"<b>Power/Type:</b> {c_power}<br/>"
                    f"<b>Coordinates:</b> {geom.y:.5f}, {geom.x:.5f}"
                    f"</div>"
                )
                folium.CircleMarker(
                    location=[geom.y, geom.x],
                    radius=5,
                    color="#1f618d",
                    weight=1,
                    fill=True,
                    fill_color="#3498db",
                    fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"Existing Charger: {c_name}",
                ).add_to(chargers_group)
        chargers_group.add_to(fmap)

    # 5. Candidate Siting Locations (Filtered)
    if candidates_wgs is not None and not candidates_wgs.empty:
        candidates_group = folium.FeatureGroup(name="Evaluated Candidate Sites", show=False)
        for _, row in candidates_wgs.iterrows():
            geom = row.geometry
            if geom and hasattr(geom, "y") and hasattr(geom, "x"):
                folium.CircleMarker(
                    location=[geom.y, geom.x],
                    radius=4,
                    color="#d35400",
                    weight=1,
                    fill=True,
                    fill_color="#e67e22",
                    fill_opacity=0.6,
                    tooltip="Candidate Siting Point",
                ).add_to(candidates_group)
        candidates_group.add_to(fmap)

    # 6. Coverage Buffer Rings (1500m)
    if sites_wgs is not None and not sites_wgs.empty:
        coverage_group = folium.FeatureGroup(
            name=f"Coverage Radius ({int(coverage_radius_m)}m)",
            show=True,
        )
        for _, row in sites_wgs.iterrows():
            geom = row.geometry
            if geom and hasattr(geom, "y") and hasattr(geom, "x"):
                folium.Circle(
                    location=[geom.y, geom.x],
                    radius=coverage_radius_m,
                    color="#27ae60",
                    weight=1.5,
                    fill=True,
                    fill_color="#2ecc71",
                    fill_opacity=0.15,
                    tooltip=f"Coverage Buffer: {int(coverage_radius_m)}m",
                ).add_to(coverage_group)
        coverage_group.add_to(fmap)

    # 7. Recommended New EV Chargers (Optimal)
    if sites_wgs is not None and not sites_wgs.empty:
        recommended_group = folium.FeatureGroup(
            name="Recommended EV Chargers (Optimal)",
            show=True,
        )
        for rank, (_, row) in enumerate(sites_wgs.iterrows(), start=1):
            geom = row.geometry
            if geom and hasattr(geom, "y") and hasattr(geom, "x"):
                site_id = row.get("site_id", f"OPT-{rank:02d}")
                dist_sub = row.get("dist_to_substation_m", row.get("substation_dist", 0.0))
                demand_cov = row.get("covered_demand", row.get("demand_score", 0.0))

                popup_card = f"""
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; min-width: 210px;">
                    <div style="background-color: #27ae60; color: white; padding: 6px 10px; border-radius: 4px 4px 0 0; font-weight: bold; font-size: 13px;">
                        ⚡ Recommended Site #{rank}
                    </div>
                    <div style="padding: 8px; font-size: 12px; background: #ffffff; border: 1px solid #dcdde1; border-top: none; border-radius: 0 0 4px 4px;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr><td style="padding: 2px 0; color: #7f8c8d;"><b>Site ID:</b></td><td style="text-align: right;">{site_id}</td></tr>
                            <tr><td style="padding: 2px 0; color: #7f8c8d;"><b>Latitude:</b></td><td style="text-align: right;">{geom.y:.5f}</td></tr>
                            <tr><td style="padding: 2px 0; color: #7f8c8d;"><b>Longitude:</b></td><td style="text-align: right;">{geom.x:.5f}</td></tr>
                            <tr><td style="padding: 2px 0; color: #7f8c8d;"><b>Substation Dist:</b></td><td style="text-align: right;">{float(dist_sub):.0f} m</td></tr>
                            <tr><td style="padding: 2px 0; color: #7f8c8d;"><b>Coverage Score:</b></td><td style="text-align: right;">{float(demand_cov):.2f}</td></tr>
                            <tr><td colspan="2" style="padding-top: 6px; text-align: center;"><span style="background: #e8f8f5; color: #27ae60; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold;">Grid-Feasible & Optimal</span></td></tr>
                        </table>
                    </div>
                </div>
                """
                folium.Marker(
                    location=[geom.y, geom.x],
                    icon=folium.Icon(color="green", icon="flash", prefix="fa"),
                    popup=folium.Popup(popup_card, max_width=280),
                    tooltip=f"Recommended Site #{rank}: {site_id}",
                ).add_to(recommended_group)
        recommended_group.add_to(fmap)

    # 8. Floating HTML Legend Box
    n_sites = len(sites_wgs) if sites_wgs is not None else 0
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        width: 270px;
        background-color: rgba(255, 255, 255, 0.96);
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        padding: 12px 16px;
        font-size: 12px;
        z-index: 9999;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        border-left: 4px solid #27ae60;
    ">
        <div style="font-weight: bold; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; color: #2c3e50;">
            ⚡ EV Charger Siting Map
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #27ae60; border-radius: 50%; margin-right: 8px;"></span>
            <span><b>Recommended Sites</b> ({n_sites} optimal)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #3498db; border-radius: 50%; margin-right: 8px;"></span>
            <span>Existing EV Chargers</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #e74c3c; border-radius: 50%; margin-right: 8px;"></span>
            <span>Power Substations</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display: inline-block; width: 14px; height: 14px; border: 2px solid #27ae60; background: rgba(46, 204, 113, 0.25); border-radius: 50%; margin-right: 8px;"></span>
            <span>Coverage Radius ({int(coverage_radius_m)}m)</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background: linear-gradient(to right, #ffffb2, #fd8d3c, #bd0026); border-radius: 2px; margin-right: 8px;"></span>
            <span>Demand / Supply Gap Hexes</span>
        </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Add Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

    # Save to disk
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_file))
    logger.info("Successfully saved interactive Folium map to %s", out_file)
    return fmap


def _calculate_fallback_metrics(
    grid_gdf: gpd.GeoDataFrame,
    selected_sites: gpd.GeoDataFrame,
    existing_chargers: Optional[gpd.GeoDataFrame],
    coverage_radius_m: float = 1500.0,
) -> Dict[str, Any]:
    """Compute vectorized coverage and network accessibility metrics.

    Used when src.metrics is unavailable or to cross-verify key metrics.

    Args:
        grid_gdf: Demand grid with weights.
        selected_sites: Newly placed charging stations.
        existing_chargers: Existing charging station locations.
        coverage_radius_m: Siting coverage radius in meters.

    Returns:
        Dict[str, Any]: Key coverage and distance performance metrics.
    """
    if grid_gdf.empty:
        return {}

    score_col = "supply_gap" if "supply_gap" in grid_gdf.columns else "demand_score"
    weights = grid_gdf[score_col].fillna(0.0).to_numpy()
    total_weight = float(weights.sum()) if weights.sum() > 0 else 1.0

    # Represent demand cells as centroids for vectorized distance calculation
    demand_pts = gpd.GeoDataFrame(
        geometry=grid_gdf.geometry.centroid,
        crs=grid_gdf.crs,
    )

    # 1. Existing coverage & distances
    has_existing = existing_chargers is not None and not existing_chargers.empty
    if has_existing:
        dist_exist = _compute_point_distances(demand_pts, existing_chargers)
        min_dist_before = np.min(dist_exist, axis=1)
        covered_before_mask = min_dist_before <= coverage_radius_m
        coverage_before = float(weights[covered_before_mask].sum() / total_weight)
        avg_dist_before = float(np.average(min_dist_before, weights=weights))
    else:
        dist_exist = np.full((len(demand_pts), 1), 10000.0)
        min_dist_before = np.full(len(demand_pts), 10000.0)
        coverage_before = 0.0
        avg_dist_before = 10000.0

    # 2. Coverage & distances with recommended sites added
    if not selected_sites.empty:
        dist_new = _compute_point_distances(demand_pts, selected_sites)
        min_dist_new = np.min(dist_new, axis=1)
        if has_existing:
            min_dist_after = np.minimum(min_dist_before, min_dist_new)
        else:
            min_dist_after = min_dist_new

        covered_after_mask = min_dist_after <= coverage_radius_m
        coverage_after = float(weights[covered_after_mask].sum() / total_weight)
        avg_dist_after = float(np.average(min_dist_after, weights=weights))
    else:
        coverage_after = coverage_before
        avg_dist_after = avg_dist_before

    dist_improvement = max(0.0, avg_dist_before - avg_dist_after)
    dist_improvement_pct = (
        (dist_improvement / avg_dist_before * 100.0) if avg_dist_before > 0 else 0.0
    )

    return {
        "demand_covered_before": coverage_before,
        "demand_covered_after": coverage_after,
        "demand_coverage_gain": max(0.0, coverage_after - coverage_before),
        "demand_coverage_gain_pct": (
            ((coverage_after - coverage_before) / max(0.001, coverage_before) * 100.0)
            if coverage_before > 0
            else 100.0
        ),
        "avg_distance_before_m": avg_dist_before,
        "avg_distance_after_m": avg_dist_after,
        "avg_distance_improvement_m": dist_improvement,
        "avg_distance_improvement_pct": dist_improvement_pct,
        "n_recommended_sites": len(selected_sites),
        "n_existing_chargers": len(existing_chargers) if has_existing else 0,
    }


def log_pipeline_summary(
    metrics: Dict[str, Any],
    selected_sites: gpd.GeoDataFrame,
    proxies_used: List[str],
    elapsed_time_s: float,
) -> None:
    """Log an executive summary of siting performance and pipeline execution.

    Args:
        metrics: Computed performance and coverage metrics.
        selected_sites: Selected optimal charging station locations.
        proxies_used: List of data proxies or fallbacks utilized.
        elapsed_time_s: Total execution wall-clock time in seconds.
    """
    cov_before = metrics.get("demand_covered_before", 0.0) * 100.0
    cov_after = metrics.get("demand_covered_after", 0.0) * 100.0
    cov_gain = metrics.get("demand_coverage_gain", 0.0) * 100.0
    dist_before = metrics.get("avg_distance_before_m", 0.0)
    dist_after = metrics.get("avg_distance_after_m", 0.0)
    dist_impr_m = metrics.get("avg_distance_improvement_m", 0.0)
    dist_impr_pct = metrics.get("avg_distance_improvement_pct", 0.0)
    n_sites = len(selected_sites)

    summary_lines = [
        "",
        "=" * 78,
        "               GRID-AWARE EV CHARGER SITING PIPELINE SUMMARY",
        "=" * 78,
        f"  Total Pipeline Runtime     : {elapsed_time_s:.2f} seconds",
        f"  Recommended Charger Sites  : {n_sites} locations selected",
        "-" * 78,
        "  KEY COVERAGE & ACCESSIBILITY METRICS:",
        f"    • Demand Covered (Before): {cov_before:6.2f}% (Existing charging network)",
        f"    • Demand Covered (After) : {cov_after:6.2f}% (Existing + Recommended)",
        f"    • Coverage Improvement   : +{cov_gain:5.2f}% absolute demand gain",
        f"    • Avg Distance to Charger: {dist_before:,.0f} m -> {dist_after:,.0f} m",
        f"    • Distance Improvement   : -{dist_impr_m:,.0f} m ({dist_impr_pct:.1f}% reduction)",
        "-" * 78,
        "  PROXIES & FALLBACKS UTILIZED:",
    ]

    if proxies_used:
        for proxy in proxies_used:
            summary_lines.append(f"    • {proxy}")
    else:
        summary_lines.append("    • None (All primary datasets loaded successfully)")

    summary_lines.extend(
        [
            "-" * 78,
            "  OUTPUT ARTIFACTS GENERATED:",
            "    • Interactive Web Map    : outputs/map.html",
            "    • Recommended Sites      : outputs/selected_sites.geojson",
            "    • Candidate Locations    : outputs/candidate_sites.geojson",
            "    • Demand Grid GeoJSON    : outputs/demand_grid.geojson",
            "    • Pipeline Metrics JSON  : outputs/metrics.json",
            "=" * 78,
            "",
        ]
    )

    for line in summary_lines:
        logger.info(line)


def run_pipeline(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the complete end-to-end EV charger siting pipeline.

    Orchestrates all pipeline stages in sequence:
      1. Load config from config.yaml
      2. Configure logging and create output directories
      3. Ingest raw OSM & grid data
      4. Preprocess and align spatial reference frames
      5. Construct H3 grid and extract features
      6. Compute demand scoring and unmet supply gap
      7. Cluster demand and generate candidate sites
      8. Apply electrical substation & spacing feasibility filters
      9. Solve MCLP optimization for optimal sites
      10. Perform model validation (holdout, baseline, sensitivity)
      11. Compute performance metrics and export GeoJSON artifacts
      12. Generate interactive Folium map

    Args:
        config_path: Optional path to custom configuration YAML.

    Returns:
        Dict[str, Any]: Execution results containing metrics, sites, and file paths.
    """
    total_start_time = time.perf_counter()
    base_dir = Path(__file__).resolve().parent

    # Step 1: Load configuration
    config = load_config(config_path)

    # Step 2: Set up logging
    log_file = base_dir / "outputs" / "pipeline.log"
    setup_logging(log_file=log_file, level=logging.INFO)
    logger.info("Starting Grid-Aware EV Charger Placement Pipeline...")
    logger.info("Target study region: %s", config.get("city", "Austin, Texas, USA"))

    # Step 3: Create output directories
    dirs = create_directories(base_dir)
    outputs_dir = dirs["outputs"]

    proxies_used: List[str] = []

    # -------------------------------------------------------------------------
    # Step a: Data Ingestion
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    logger.info("Stage 1/10: Ingesting raw geospatial and grid network data...")
    try:
        raw_data = _invoke_step(ingest_data, config=config, func_name="ingest_data")
        if isinstance(raw_data, dict):
            proxies_used.extend(raw_data.get("proxies_used", []))
        logger.info("Stage 1/10 completed in %.2f s", time.perf_counter() - step_start)
    except Exception as e:
        logger.error("Stage 1/10 (Ingestion) encountered a fatal error: %s", e, exc_info=True)
        raise

    # -------------------------------------------------------------------------
    # Step b: Preprocessing
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    logger.info("Stage 2/10: Preprocessing geospatial layers and computing weights...")
    try:
        preprocessed_data = _invoke_step(
            preprocess_data,
            raw_data=raw_data,
            config=config,
            func_name="preprocess_data",
        )
        if isinstance(preprocessed_data, dict):
            proxies_used.extend(preprocessed_data.get("proxies_used", []))
        logger.info("Stage 2/10 completed in %.2f s", time.perf_counter() - step_start)
    except Exception as e:
        logger.error("Stage 2/10 (Preprocessing) encountered a fatal error: %s", e, exc_info=True)
        raise

    boundary = _extract_layer(preprocessed_data, "boundary", "boundary_gdf")
    existing_chargers = _extract_layer(
        preprocessed_data, "existing_chargers", "chargers_gdf", "chargers"
    )
    substations = _extract_layer(preprocessed_data, "substations", "substations_gdf")

    # -------------------------------------------------------------------------
    # Step c: Grid & Cell Features
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    h3_res = int(config.get("h3_resolution", 8))
    logger.info("Stage 3/10: Generating H3 grid (res=%d) and computing cell features...", h3_res)
    try:
        grid_gdf = _invoke_step(
            create_h3_grid,
            boundary=boundary,
            resolution=h3_res,
            config=config,
            preprocessed_data=preprocessed_data,
            func_name="create_h3_grid",
        )
        grid_features_gdf = _invoke_step(
            compute_cell_features,
            grid_gdf=grid_gdf,
            preprocessed_data=preprocessed_data,
            config=config,
            func_name="compute_cell_features",
        )
        logger.info(
            "Stage 3/10 completed in %.2f s (%d H3 cells processed)",
            time.perf_counter() - step_start,
            len(grid_features_gdf),
        )
    except Exception as e:
        logger.error("Stage 3/10 (Grid & Features) encountered a fatal error: %s", e, exc_info=True)
        raise

    # -------------------------------------------------------------------------
    # Step d: Demand Scoring & Supply Gap
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    logger.info("Stage 4/10: Calculating demand scores and unmet supply gaps...")
    try:
        scored_grid = _invoke_step(
            compute_demand_score,
            grid_features_gdf=grid_features_gdf,
            grid_gdf=grid_features_gdf,
            config=config,
            func_name="compute_demand_score",
        )
        supply_gap_grid = _invoke_step(
            compute_supply_gap,
            grid_gdf=scored_grid,
            preprocessed_data=preprocessed_data,
            config=config,
            func_name="compute_supply_gap",
        )

        # Save standalone demand heatmap
        heatmap_path = outputs_dir / "demand_heatmap.html"
        try:
            if save_demand_heatmap is not None:
                _invoke_step(
                    save_demand_heatmap,
                    grid_gdf=supply_gap_grid,
                    output_path=heatmap_path,
                    config=config,
                    func_name="save_demand_heatmap",
                )
                logger.info("Demand heatmap saved to %s", heatmap_path)
        except Exception as hm_err:
            logger.warning("Demand heatmap generation skipped: %s", hm_err)

        logger.info("Stage 4/10 completed in %.2f s", time.perf_counter() - step_start)
    except Exception as e:
        logger.error("Stage 4/10 (Scoring) encountered a fatal error: %s", e, exc_info=True)
        raise

    # -------------------------------------------------------------------------
    # Step e: Clustering & Candidate Site Extraction
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    logger.info("Stage 5/10: Clustering demand hotspots and extracting candidate sites...")
    try:
        clustered_cells = _invoke_step(
            cluster_demand_cells,
            grid_gdf=supply_gap_grid,
            config=config,
            func_name="cluster_demand_cells",
        )
        candidate_sites = _invoke_step(
            get_candidate_sites,
            clustered_cells_gdf=clustered_cells,
            grid_gdf=supply_gap_grid,
            config=config,
            preprocessed_data=preprocessed_data,
            func_name="get_candidate_sites",
        )
        logger.info(
            "Stage 5/10 completed in %.2f s (%d candidate sites identified)",
            time.perf_counter() - step_start,
            len(candidate_sites),
        )
    except Exception as e:
        logger.error("Stage 5/10 (Clustering) encountered a fatal error: %s", e, exc_info=True)
        raise

    # -------------------------------------------------------------------------
    # Step f: Feasibility Filter (Substation distance + Spacing)
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    max_sub_dist = float(config.get("max_substation_dist_m", 3000.0))
    min_spacing = float(config.get("min_spacing_m", 800.0))
    logger.info(
        "Stage 6/10: Filtering feasibility (substation dist <= %.0fm, spacing >= %.0fm)...",
        max_sub_dist,
        min_spacing,
    )
    try:
        feasible_candidates = _invoke_step(
            feasibility_filter,
            candidate_sites=candidate_sites,
            substations=substations,
            existing_chargers=existing_chargers,
            config=config,
            preprocessed_data=preprocessed_data,
            func_name="feasibility_filter",
        )
        logger.info(
            "Stage 6/10 completed in %.2f s (%d / %d candidates passed)",
            time.perf_counter() - step_start,
            len(feasible_candidates),
            len(candidate_sites),
        )
    except Exception as e:
        logger.error("Stage 6/10 (Feasibility) encountered a fatal error: %s", e, exc_info=True)
        raise

    # -------------------------------------------------------------------------
    # Step g: Optimization (MCLP)
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    n_new_stations = int(config.get("n_new_stations", 20))
    cov_radius = float(config.get("coverage_radius_m", 1500.0))
    solver_timeout = int(config.get("solver_time_limit_s", 60))

    logger.info(
        "Stage 7/10: Solving MCLP optimization (p=%d stations, radius=%.0fm, timeout=%ds)...",
        n_new_stations,
        cov_radius,
        solver_timeout,
    )
    try:
        opt_output = _invoke_step(
            solve_mclp,
            demand_grid=supply_gap_grid,
            candidate_sites=feasible_candidates,
            config=config,
            n_new_stations=n_new_stations,
            coverage_radius_m=cov_radius,
            min_spacing_m=min_spacing,
            solver_time_limit_s=solver_timeout,
            func_name="solve_mclp",
        )

        if isinstance(opt_output, gpd.GeoDataFrame):
            selected_sites = opt_output
        elif isinstance(opt_output, dict) and "selected_sites" in opt_output:
            selected_sites = opt_output["selected_sites"]
        elif isinstance(opt_output, tuple) and len(opt_output) > 0 and isinstance(opt_output[0], gpd.GeoDataFrame):
            selected_sites = opt_output[0]
        else:
            logger.warning("Unexpected solver return type. Casting to GeoDataFrame if possible.")
            selected_sites = gpd.GeoDataFrame(opt_output)

        logger.info(
            "Stage 7/10 completed in %.2f s (%d optimal sites selected)",
            time.perf_counter() - step_start,
            len(selected_sites),
        )
    except Exception as e:
        logger.error("Stage 7/10 (Optimization) encountered a fatal error: %s", e, exc_info=True)
        raise

    # -------------------------------------------------------------------------
    # Step h: Validation (Holdout, Baselines, Sensitivity)
    # Requirement: Handle errors per step - if validation fails, still save main outputs
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    logger.info("Stage 8/10: Running model validation suite...")
    validation_results: Dict[str, Any] = {}

    # Holdout Validation
    try:
        if holdout_validation is not None:
            logger.info("Running holdout cross-validation...")
            validation_results["holdout"] = _invoke_step(
                holdout_validation,
                selected_sites=selected_sites,
                preprocessed_data=preprocessed_data,
                config=config,
                demand_grid=supply_gap_grid,
                func_name="holdout_validation",
            )
    except Exception as v_err:
        logger.warning("Holdout validation failed (continuing pipeline): %s", v_err)
        validation_results["holdout"] = {"status": "failed", "error": str(v_err)}

    # Baseline Comparison (Random & Heuristic)
    try:
        if baseline_comparison is not None:
            logger.info("Running baseline benchmarking...")
            validation_results["baseline"] = _invoke_step(
                baseline_comparison,
                selected_sites=selected_sites,
                demand_grid=supply_gap_grid,
                candidate_sites=feasible_candidates,
                config=config,
                func_name="baseline_comparison",
            )
    except Exception as v_err:
        logger.warning("Baseline comparison failed (continuing pipeline): %s", v_err)
        validation_results["baseline"] = {"status": "failed", "error": str(v_err)}

    # Sensitivity Analysis
    try:
        if sensitivity_analysis is not None:
            logger.info("Running sensitivity analysis...")
            validation_results["sensitivity"] = _invoke_step(
                sensitivity_analysis,
                demand_grid=supply_gap_grid,
                candidate_sites=feasible_candidates,
                config=config,
                func_name="sensitivity_analysis",
            )
    except Exception as v_err:
        logger.warning("Sensitivity analysis failed (continuing pipeline): %s", v_err)
        validation_results["sensitivity"] = {"status": "failed", "error": str(v_err)}

    logger.info("Stage 8/10 completed in %.2f s", time.perf_counter() - step_start)

    # -------------------------------------------------------------------------
    # Step i: Metrics Calculation & Export
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    logger.info("Stage 9/10: Computing comprehensive evaluation metrics...")
    metrics: Dict[str, Any] = {}
    try:
        if compute_metrics is not None:
            metrics = _invoke_step(
                compute_metrics,
                grid_gdf=supply_gap_grid,
                selected_sites=selected_sites,
                preprocessed_data=preprocessed_data,
                config=config,
                validation_results=validation_results,
                func_name="compute_metrics",
            )
    except Exception as m_err:
        logger.warning("compute_metrics raised an error (%s). Falling back to internal metric engine.", m_err)

    # Cross-verify or populate fallback metrics if needed
    fallback_metrics = _calculate_fallback_metrics(
        grid_gdf=supply_gap_grid,
        selected_sites=selected_sites,
        existing_chargers=existing_chargers,
        coverage_radius_m=cov_radius,
    )
    for k, v in fallback_metrics.items():
        if k not in metrics or metrics[k] is None:
            metrics[k] = v

    metrics["validation"] = validation_results
    metrics["execution_time_s"] = round(time.perf_counter() - total_start_time, 2)
    metrics["proxies_used"] = list(set(proxies_used))

    # Save metrics JSON artifact
    metrics_path = outputs_dir / "metrics.json"
    try:
        if save_metrics is not None:
            _invoke_step(
                save_metrics,
                metrics=metrics,
                output_path=metrics_path,
                func_name="save_metrics",
            )
        else:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)
        logger.info("Saved evaluation metrics to %s", metrics_path)
    except Exception as sm_err:
        logger.error("Failed saving metrics to %s: %s", metrics_path, sm_err)

    # -------------------------------------------------------------------------
    # Export intermediate geospatial artifacts
    # -------------------------------------------------------------------------
    try:
        if not selected_sites.empty:
            sel_out = _ensure_wgs84(selected_sites)
            sel_out.to_file(outputs_dir / "selected_sites.geojson", driver="GeoJSON")
        if not feasible_candidates.empty:
            cand_out = _ensure_wgs84(feasible_candidates)
            cand_out.to_file(outputs_dir / "candidate_sites.geojson", driver="GeoJSON")
        if not supply_gap_grid.empty:
            grid_out = _ensure_wgs84(supply_gap_grid)
            grid_out.to_file(outputs_dir / "demand_grid.geojson", driver="GeoJSON")
    except Exception as exp_err:
        logger.warning("Exporting intermediate GeoJSON files encountered an issue: %s", exp_err)

    # -------------------------------------------------------------------------
    # Step j: Save Map Artifact (Folium)
    # -------------------------------------------------------------------------
    step_start = time.perf_counter()
    logger.info("Stage 10/10: Building interactive Folium map artifact...")
    map_path = outputs_dir / "map.html"
    try:
        save_map(
            grid_gdf=supply_gap_grid,
            selected_sites=selected_sites,
            candidate_sites=feasible_candidates,
            existing_chargers=existing_chargers,
            substations=substations,
            boundary=boundary,
            config=config,
            output_path=map_path,
        )
        logger.info("Stage 10/10 completed in %.2f s", time.perf_counter() - step_start)
    except Exception as map_err:
        logger.error("Stage 10/10 (Map Generation) encountered an error: %s", map_err, exc_info=True)

    # -------------------------------------------------------------------------
    # Executive Summary Log
    # -------------------------------------------------------------------------
    total_elapsed = time.perf_counter() - total_start_time
    log_pipeline_summary(
        metrics=metrics,
        selected_sites=selected_sites,
        proxies_used=list(set(proxies_used)),
        elapsed_time_s=total_elapsed,
    )

    return {
        "config": config,
        "selected_sites": selected_sites,
        "candidate_sites": feasible_candidates,
        "grid_gdf": supply_gap_grid,
        "metrics": metrics,
        "validation": validation_results,
        "map_path": map_path,
        "proxies_used": list(set(proxies_used)),
        "elapsed_time_s": total_elapsed,
    }


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as err:
        logger.fatal("Pipeline terminated unexpectedly: %s", err, exc_info=True)
        sys.exit(1)
