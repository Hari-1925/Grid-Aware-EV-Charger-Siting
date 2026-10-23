"""
H3 Grid Generation and Feature Engineering Module.

This module provides functions to:
1. Construct a discrete hexagonal spatial grid (Uber H3) over an urban boundary.
2. Enrich each H3 grid cell with multi-modal urban features:
   - Population proxy (building counts or raster population sum)
   - POI counts and category diversity
   - Traffic intensity proxy (weighted road network density)
   - Distance to nearest existing EV charger and charger density
   - Distance to nearest electrical substation
3. Min-max normalize all computed features to [0, 1].
4. Export the feature-enriched grid to GeoPackage format for optimization and siting models.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon, mapping

try:
    import h3
except ImportError:
    h3 = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

__all__ = [
    "create_h3_grid",
    "compute_cell_features",
    "min_max_normalize",
]


# =====================================================================
# H3 Compatibility Helpers (supporting H3 v3 and v4 APIs)
# =====================================================================

def _polygon_to_h3_cells(poly: Polygon, resolution: int) -> set:
    """Cover a single Shapely Polygon with H3 cells at the specified resolution.

    Tries H3 v4 `geo_to_cells`, `polygon_to_cells`, and H3 v3 `polyfill_geojson`/`polyfill`.
    Falls back to the polygon centroid if polyfill yields an empty set.

    Args:
        poly: A Shapely Polygon geometry in WGS84 (EPSG:4326).
        resolution: H3 resolution index (e.g., 7, 8, 9).

    Returns:
        Set of H3 cell index strings covering the polygon.
    """
    if h3 is None:
        raise ImportError(
            "The 'h3' package is required but not installed. "
            "Please install it using 'pip install h3'."
        )

    if poly.is_empty or not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty:
            return set()

    cells: set = set()

    # Attempt 1: H3 v4 geo_to_cells (expects GeoJSON dict)
    if hasattr(h3, "geo_to_cells"):
        try:
            geo_dict = mapping(poly)
            res = h3.geo_to_cells(geo_dict, resolution)
            if res:
                return set(res)
        except Exception as err:
            logger.debug(f"h3.geo_to_cells attempt failed: {err}")

    # Attempt 2: H3 v4 polygon_to_cells with h3.Polygon
    if hasattr(h3, "polygon_to_cells"):
        try:
            coords = list(poly.exterior.coords)
            # Try passing (lat, lng) pairs
            latlng_coords = [(lat, lng) for lng, lat in coords]
            try:
                if hasattr(h3, "Polygon"):
                    h3_poly = h3.Polygon(latlng_coords)
                    res = h3.polygon_to_cells(h3_poly, resolution)
                    if res:
                        return set(res)
            except Exception:
                pass

            # Try passing standard (lng, lat)
            try:
                if hasattr(h3, "Polygon"):
                    h3_poly = h3.Polygon(coords)
                    res = h3.polygon_to_cells(h3_poly, resolution)
                    if res:
                        return set(res)
            except Exception:
                pass

            # Try passing GeoJSON mapping directly
            try:
                res = h3.polygon_to_cells(mapping(poly), resolution)
                if res:
                    return set(res)
            except Exception:
                pass
        except Exception as err:
            logger.debug(f"h3.polygon_to_cells attempt failed: {err}")

    # Attempt 3: H3 v3 polyfill_geojson
    if hasattr(h3, "polyfill_geojson"):
        try:
            res = h3.polyfill_geojson(mapping(poly), resolution)
            if res:
                return set(res)
        except Exception as err:
            logger.debug(f"h3.polyfill_geojson attempt failed: {err}")

    # Attempt 4: H3 v3 polyfill
    if hasattr(h3, "polyfill"):
        try:
            res = h3.polyfill(mapping(poly), resolution, geo_json_conformant=True)
            if res:
                return set(res)
        except Exception as err:
            logger.debug(f"h3.polyfill attempt failed: {err}")

    # Fallback: Small polygon whose centroid falls inside an H3 cell
    centroid = poly.centroid
    lat, lng = centroid.y, centroid.x
    fallback_cell = None
    if hasattr(h3, "latlng_to_cell"):
        fallback_cell = h3.latlng_to_cell(lat, lng, resolution)
    elif hasattr(h3, "geo_to_h3"):
        fallback_cell = h3.geo_to_h3(lat, lng, resolution)

    if fallback_cell:
        return {fallback_cell}

    return set()


def _get_cells_for_geometry(geom: Any, resolution: int) -> set:
    """Extract all H3 cell indices covering a Polygon or MultiPolygon.

    Args:
        geom: A Shapely Polygon, MultiPolygon, or GeometryCollection.
        resolution: H3 resolution index.

    Returns:
        Set of H3 index strings.
    """
    cells = set()
    if geom is None or geom.is_empty:
        return cells

    if isinstance(geom, Polygon):
        cells.update(_polygon_to_h3_cells(geom, resolution))
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            cells.update(_polygon_to_h3_cells(part, resolution))
    elif hasattr(geom, "geoms"):
        for sub_geom in geom.geoms:
            cells.update(_get_cells_for_geometry(sub_geom, resolution))
    return cells


def _cell_to_polygon(cell: str) -> Polygon:
    """Convert an H3 cell index into a Shapely Polygon in WGS84 (EPSG:4326).

    Args:
        cell: H3 cell index string.

    Returns:
        Shapely Polygon representing the hexagonal cell boundary.
    """
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(cell)
    elif hasattr(h3, "h3_to_geo_boundary"):
        boundary = h3.h3_to_geo_boundary(cell, geo_json=False)
    else:
        raise AttributeError("No boundary function found in h3 module")

    # H3 returns boundary as (lat, lng) pairs; Shapely expects (lng, lat)
    coords = [(lng, lat) for lat, lng in boundary]
    return Polygon(coords)


def _cell_to_centroid_coords(cell: str) -> Tuple[float, float]:
    """Retrieve centroid coordinates (lat, lng) for an H3 cell index.

    Args:
        cell: H3 cell index string.

    Returns:
        Tuple of (latitude, longitude) floats.
    """
    if hasattr(h3, "cell_to_latlng"):
        return tuple(h3.cell_to_latlng(cell))
    elif hasattr(h3, "h3_to_geo"):
        return tuple(h3.h3_to_geo(cell))
    else:
        raise AttributeError("No centroid function found in h3 module")


# =====================================================================
# H3 Grid Creation
# =====================================================================

def create_h3_grid(
    boundary_gdf: gpd.GeoDataFrame,
    resolution: int,
    utm_crs: Any
) -> gpd.GeoDataFrame:
    """Generate an H3 hexagonal grid over a study area boundary and project to UTM.

    Performs the following steps:
    1. Reprojects boundary geometry to WGS84 (EPSG:4326).
    2. Uses H3 spatial indexing to identify all hexagonal cells covering the area.
    3. Builds Shapely Polygons for each cell boundary.
    4. Computes centroid coordinates (centroid_lat, centroid_lon) and Point geometry.
    5. Projects the resulting GeoDataFrame and centroid points to the target UTM CRS.

    Args:
        boundary_gdf: GeoDataFrame containing study area boundary geometry.
        resolution: H3 resolution level (typically 7 to 9 for urban analysis).
        utm_crs: Target projected coordinate reference system (e.g., "EPSG:32614").

    Returns:
        A GeoDataFrame projected to `utm_crs` with columns:
        - h3_index: Unique H3 cell identifier string.
        - geometry: Hexagon polygon in target UTM CRS.
        - centroid_lat: Latitude of cell centroid (WGS84).
        - centroid_lon: Longitude of cell centroid (WGS84).
        - centroid_geometry: Point geometry of cell centroid in target UTM CRS.

    Raises:
        ValueError: If `boundary_gdf` is empty or no H3 cells could be generated.
    """
    if boundary_gdf is None or boundary_gdf.empty:
        raise ValueError("boundary_gdf cannot be empty or None")

    logger.info(
        f"Creating H3 grid at resolution {resolution} for boundary "
        f"with {len(boundary_gdf)} row(s)..."
    )

    try:
        # Step 1: Ensure boundary is in WGS84
        if boundary_gdf.crs != "EPSG:4326":
            boundary_wgs84 = boundary_gdf.to_crs("EPSG:4326")
        else:
            boundary_wgs84 = boundary_gdf.copy()

        # Step 2: Merge geometries to handle multi-part boundaries
        merged_geom = boundary_wgs84.geometry.unary_union
        if merged_geom.is_empty:
            raise ValueError("Merged boundary geometry is empty")

        # Step 3: Polyfill with H3 cells
        cells = _get_cells_for_geometry(merged_geom, resolution)
        if not cells:
            raise ValueError(
                f"No H3 cells could be generated at resolution {resolution} "
                "for the provided boundary."
            )

        cells_sorted = sorted(list(cells))
        logger.info(f"Generated {len(cells_sorted)} H3 cells covering boundary.")

        # Step 4: Construct geometry and centroid records
        polygons: List[Polygon] = []
        lats: List[float] = []
        lons: List[float] = []
        centroid_points: List[Point] = []

        for cell in cells_sorted:
            poly = _cell_to_polygon(cell)
            lat, lon = _cell_to_centroid_coords(cell)
            polygons.append(poly)
            lats.append(lat)
            lons.append(lon)
            centroid_points.append(Point(lon, lat))

        # Build WGS84 GeoDataFrame
        grid_wgs84 = gpd.GeoDataFrame(
            {
                "h3_index": cells_sorted,
                "geometry": polygons,
                "centroid_lat": lats,
                "centroid_lon": lons,
                "centroid_geometry": centroid_points,
            },
            crs="EPSG:4326"
        )

        # Step 5: Reproject to target UTM CRS
        grid_utm = grid_wgs84.to_crs(utm_crs)
        # Reproject secondary centroid geometry series to match UTM CRS
        grid_utm["centroid_geometry"] = gpd.GeoSeries(
            grid_utm["centroid_geometry"],
            crs="EPSG:4326"
        ).to_crs(utm_crs)

        logger.info(
            f"Successfully built H3 grid with {len(grid_utm)} cells in CRS {utm_crs}."
        )
        return grid_utm

    except Exception as err:
        logger.error(f"Failed to create H3 grid: {err}", exc_info=True)
        raise


# =====================================================================
# Feature Extraction and Engineering
# =====================================================================

def min_max_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric pandas Series to the range [0, 1].

    Handles constant or zero-variance series gracefully by returning zeros.

    Args:
        series: Pandas Series with numeric values.

    Returns:
        Series normalized to [0, 1].
    """
    if series.empty:
        return series.astype(float)

    s_clean = series.fillna(0.0).astype(float)
    min_val = s_clean.min()
    max_val = s_clean.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val <= min_val:
        return pd.Series(0.0, index=series.index, dtype=float)

    return (s_clean - min_val) / (max_val - min_val)


def _load_layer(
    processed_paths: Dict[str, Any],
    candidate_keys: List[str],
    utm_crs: Any
) -> Optional[gpd.GeoDataFrame]:
    """Helper to retrieve and reproject a vector geospatial layer.

    Accepts either an existing GeoDataFrame or a file path (GPKG, SHP, GeoJSON, Parquet).

    Args:
        processed_paths: Dictionary mapping layer names to paths or GeoDataFrames.
        candidate_keys: Candidate key names to look up in `processed_paths`.
        utm_crs: Target UTM CRS for reprojection.

    Returns:
        Projected GeoDataFrame or None if not found/empty.
    """
    for key in candidate_keys:
        if key in processed_paths and processed_paths[key] is not None:
            val = processed_paths[key]
            try:
                # Case 1: In-memory GeoDataFrame
                if isinstance(val, gpd.GeoDataFrame):
                    if val.empty:
                        return None
                    if val.crs is None:
                        val = val.set_crs(utm_crs)
                    elif val.crs != utm_crs:
                        val = val.to_crs(utm_crs)
                    return val

                # Case 2: In-memory pandas DataFrame with lat/lon
                if isinstance(val, pd.DataFrame):
                    if val.empty:
                        return None
                    lon_col = next((c for c in ["lon", "lng", "longitude"] if c in val.columns), None)
                    lat_col = next((c for c in ["lat", "latitude"] if c in val.columns), None)
                    if lon_col and lat_col:
                        gdf = gpd.GeoDataFrame(
                            val,
                            geometry=gpd.points_from_xy(val[lon_col], val[lat_col]),
                            crs="EPSG:4326"
                        ).to_crs(utm_crs)
                        return gdf
                    return None

                # Case 3: File path
                path = Path(val)
                if path.exists() and path.is_file():
                    gdf = gpd.read_file(path)
                    if gdf.empty:
                        return None
                    if gdf.crs is None:
                        gdf = gdf.set_crs(utm_crs)
                    elif gdf.crs != utm_crs:
                        gdf = gdf.to_crs(utm_crs)
                    return gdf
                else:
                    logger.debug(f"Path for layer key '{key}' does not exist: {path}")

            except Exception as err:
                logger.warning(f"Error loading layer for key '{key}': {err}")

    return None


def _compute_population_proxy(
    grid_gdf: gpd.GeoDataFrame,
    processed_paths: Dict[str, Any],
    utm_crs: Any
) -> pd.Series:
    """Compute per-cell population proxy from building footprints or raster data.

    Args:
        grid_gdf: H3 grid GeoDataFrame in UTM CRS.
        processed_paths: Dictionary of processed dataset paths.
        utm_crs: Target UTM CRS.

    Returns:
        Series of population counts/sums aligned with `grid_gdf.index`.
    """
    # Option 1: Raster population sum if raster path is provided
    raster_val = (
        processed_paths.get("population_raster")
        or processed_paths.get("population")
    )
    if raster_val and isinstance(raster_val, (str, Path)):
        raster_path = Path(raster_val)
        if raster_path.exists() and raster_path.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
            try:
                import rasterio
                from rasterio.mask import mask

                logger.info(f"Computing population proxy from raster: {raster_path}")
                sums: List[float] = []
                with rasterio.open(raster_path) as src:
                    # Reproject grid geometry to raster CRS if necessary
                    if grid_gdf.crs != src.crs:
                        polys_raster_crs = grid_gdf.geometry.to_crs(src.crs)
                    else:
                        polys_raster_crs = grid_gdf.geometry

                    for poly in polys_raster_crs:
                        try:
                            out_image, _ = mask(src, [poly], crop=True)
                            valid_data = out_image[out_image != src.nodata]
                            valid_positive = valid_data[valid_data > 0]
                            sums.append(float(np.sum(valid_positive)))
                        except Exception:
                            sums.append(0.0)

                return pd.Series(sums, index=grid_gdf.index, dtype=float)
            except ImportError:
                logger.debug("rasterio not installed; falling back to building footprints.")
            except Exception as err:
                logger.warning(f"Raster population computation failed ({err}); using buildings.")

    # Option 2: Count of buildings within each cell
    buildings_gdf = _load_layer(
        processed_paths,
        ["buildings", "building", "structures", "footprints"],
        utm_crs
    )

    if buildings_gdf is not None and not buildings_gdf.empty:
        logger.info(f"Computing population proxy from {len(buildings_gdf)} building footprints...")
        valid_buildings = buildings_gdf[
            buildings_gdf.geometry.is_valid & ~buildings_gdf.geometry.is_empty
        ].copy()

        # Represent each building by its centroid for unambiguous cell assignment
        valid_buildings["geometry"] = valid_buildings.geometry.centroid
        joined = gpd.sjoin(
            valid_buildings[["geometry"]],
            grid_gdf[["h3_index", "geometry"]],
            how="inner",
            predicate="intersects"
        )
        counts = joined.groupby("h3_index").size()
        return grid_gdf["h3_index"].map(counts).fillna(0.0).astype(float)

    logger.warning("No building or raster population data found. Setting population_proxy to 0.0.")
    return pd.Series(0.0, index=grid_gdf.index, dtype=float)


def _compute_poi_features(
    grid_gdf: gpd.GeoDataFrame,
    processed_paths: Dict[str, Any],
    config: Dict[str, Any],
    utm_crs: Any
) -> Tuple[pd.Series, pd.Series]:
    """Compute POI count and POI category diversity per H3 cell.

    Args:
        grid_gdf: H3 grid GeoDataFrame in UTM CRS.
        processed_paths: Dictionary of processed dataset paths.
        config: Configuration dictionary.
        utm_crs: Target UTM CRS.

    Returns:
        Tuple of (poi_count_series, poi_diversity_series).
    """
    pois_gdf = _load_layer(
        processed_paths,
        ["pois", "poi", "amenities", "places", "points_of_interest"],
        utm_crs
    )

    if pois_gdf is None or pois_gdf.empty:
        logger.warning("No POI data found. Setting poi_count and poi_diversity to 0.0.")
        zero_series = pd.Series(0.0, index=grid_gdf.index, dtype=float)
        return zero_series, zero_series

    logger.info(f"Computing POI features from {len(pois_gdf)} POIs...")
    valid_pois = pois_gdf[
        pois_gdf.geometry.is_valid & ~pois_gdf.geometry.is_empty
    ].copy()

    # Convert polygon POIs to centroids
    valid_pois["geometry"] = valid_pois.geometry.centroid

    # Detect category column
    cat_col = None
    for col_candidate in ["category", "poi_category", "amenity", "shop", "type", "poi_type", "class"]:
        if col_candidate in valid_pois.columns:
            cat_col = col_candidate
            break

    if cat_col is None:
        valid_pois["category"] = "general"
        cat_col = "category"

    # Optional filter based on configured POI categories if available
    configured_cats = config.get("poi_categories")
    if configured_cats:
        cats_lower = {str(c).strip().lower() for c in configured_cats}
        matches = valid_pois[cat_col].astype(str).str.lower().isin(cats_lower)
        if matches.any():
            valid_pois = valid_pois[matches].copy()

    # Spatial join POIs to grid cells
    joined = gpd.sjoin(
        valid_pois[["geometry", cat_col]],
        grid_gdf[["h3_index", "geometry"]],
        how="inner",
        predicate="intersects"
    )

    poi_counts = joined.groupby("h3_index").size()
    poi_count_series = grid_gdf["h3_index"].map(poi_counts).fillna(0.0).astype(float)

    poi_div = joined.groupby("h3_index")[cat_col].nunique()
    poi_diversity_series = grid_gdf["h3_index"].map(poi_div).fillna(0.0).astype(float)

    return poi_count_series, poi_diversity_series


def _compute_traffic_proxy(
    grid_gdf: gpd.GeoDataFrame,
    processed_paths: Dict[str, Any],
    config: Dict[str, Any],
    utm_crs: Any
) -> pd.Series:
    """Compute traffic proxy per cell as the sum of road class weights.

    Args:
        grid_gdf: H3 grid GeoDataFrame in UTM CRS.
        processed_paths: Dictionary of processed dataset paths.
        config: Configuration dictionary containing `road_class_weights`.
        utm_crs: Target UTM CRS.

    Returns:
        Series of traffic intensity proxy values aligned with `grid_gdf.index`.
    """
    roads_gdf = _load_layer(
        processed_paths,
        ["roads", "road_network", "highways", "streets", "network"],
        utm_crs
    )

    if roads_gdf is None or roads_gdf.empty:
        logger.warning("No road network data found. Setting traffic_proxy to 0.0.")
        return pd.Series(0.0, index=grid_gdf.index, dtype=float)

    logger.info(f"Computing traffic proxy from {len(roads_gdf)} road segments...")
    valid_roads = roads_gdf[
        roads_gdf.geometry.is_valid & ~roads_gdf.geometry.is_empty
    ].copy()

    # Road class weights from configuration with robust defaults
    default_weights = {
        "motorway": 5.0,
        "motorway_link": 4.0,
        "trunk": 4.0,
        "trunk_link": 3.0,
        "primary": 3.0,
        "primary_link": 2.0,
        "secondary": 2.0,
        "secondary_link": 1.5,
        "tertiary": 1.0,
        "tertiary_link": 0.75,
        "residential": 0.5,
        "living_street": 0.5,
        "unclassified": 0.5,
    }
    class_weights = config.get("road_class_weights", default_weights)

    # Detect highway classification column
    hw_col = None
    for candidate in ["highway", "road_class", "type", "highway_type", "fclass"]:
        if candidate in valid_roads.columns:
            hw_col = candidate
            break

    def _resolve_weight(val: Any) -> float:
        if val is None or pd.isna(val):
            return 0.1
        if isinstance(val, (list, tuple, np.ndarray)):
            w_list = [class_weights.get(str(v).strip().lower(), 0.1) for v in val]
            return max(w_list) if w_list else 0.1
        return class_weights.get(str(val).strip().lower(), 0.1)

    if hw_col:
        valid_roads["road_weight"] = valid_roads[hw_col].apply(_resolve_weight)
    else:
        valid_roads["road_weight"] = 0.5

    # Spatial join road segments intersecting each cell polygon
    joined = gpd.sjoin(
        valid_roads[["geometry", "road_weight"]],
        grid_gdf[["h3_index", "geometry"]],
        how="inner",
        predicate="intersects"
    )

    traffic_sums = joined.groupby("h3_index")["road_weight"].sum()
    return grid_gdf["h3_index"].map(traffic_sums).fillna(0.0).astype(float)


def _compute_charger_features(
    grid_gdf: gpd.GeoDataFrame,
    processed_paths: Dict[str, Any],
    config: Dict[str, Any],
    utm_crs: Any
) -> Tuple[pd.Series, pd.Series]:
    """Compute distance to nearest charger (meters) and charger count within radius.

    Uses `scipy.spatial.cKDTree` in projected UTM coordinates for efficient
    nearest-neighbor and radial query evaluation.

    Args:
        grid_gdf: H3 grid GeoDataFrame in UTM CRS.
        processed_paths: Dictionary of processed dataset paths.
        config: Configuration dictionary containing `coverage_radius_m`.
        utm_crs: Target UTM CRS.

    Returns:
        Tuple of (dist_nearest_charger_series, chargers_in_radius_series).
    """
    chargers_gdf = _load_layer(
        processed_paths,
        ["chargers", "existing_chargers", "ev_chargers", "charging_stations"],
        utm_crs
    )

    # Use cell centroids in UTM coordinates (meters)
    cell_centroids = np.column_stack([
        grid_gdf.geometry.centroid.x,
        grid_gdf.geometry.centroid.y
    ])
    coverage_radius_m = float(config.get("coverage_radius_m", 1500.0))

    if chargers_gdf is None or chargers_gdf.empty:
        logger.warning(
            "No existing chargers data found. Using default large distance (50km) and zero count."
        )
        dist_series = pd.Series(50000.0, index=grid_gdf.index, dtype=float)
        count_series = pd.Series(0.0, index=grid_gdf.index, dtype=float)
        return dist_series, count_series

    logger.info(f"Computing charger spatial proximity from {len(chargers_gdf)} chargers...")
    valid_chargers = chargers_gdf[
        chargers_gdf.geometry.is_valid & ~chargers_gdf.geometry.is_empty
    ]

    if valid_chargers.empty:
        dist_series = pd.Series(50000.0, index=grid_gdf.index, dtype=float)
        count_series = pd.Series(0.0, index=grid_gdf.index, dtype=float)
        return dist_series, count_series

    charger_pts = valid_chargers.geometry.centroid
    charger_coords = np.column_stack([charger_pts.x, charger_pts.y])

    # Build spatial k-d tree
    charger_tree = cKDTree(charger_coords)

    # Query nearest charger distance
    nearest_dist, _ = charger_tree.query(cell_centroids, k=1)
    dist_series = pd.Series(nearest_dist, index=grid_gdf.index, dtype=float)

    # Query count of chargers within coverage radius
    try:
        counts = charger_tree.query_ball_point(
            cell_centroids,
            r=coverage_radius_m,
            return_length=True
        )
        count_series = pd.Series(counts, index=grid_gdf.index, dtype=float)
    except TypeError:
        # Fallback for scipy versions lacking return_length
        neighbors = charger_tree.query_ball_point(cell_centroids, r=coverage_radius_m)
        counts_list = [len(idx) for idx in neighbors]
        count_series = pd.Series(counts_list, index=grid_gdf.index, dtype=float)

    return dist_series, count_series


def _compute_substation_features(
    grid_gdf: gpd.GeoDataFrame,
    processed_paths: Dict[str, Any],
    config: Dict[str, Any],
    utm_crs: Any
) -> pd.Series:
    """Compute distance to the nearest electrical substation (meters) using cKDTree.

    Args:
        grid_gdf: H3 grid GeoDataFrame in UTM CRS.
        processed_paths: Dictionary of processed dataset paths.
        config: Configuration dictionary containing `max_substation_dist_m`.
        utm_crs: Target UTM CRS.

    Returns:
        Series of nearest substation distances in meters aligned with `grid_gdf.index`.
    """
    substations_gdf = _load_layer(
        processed_paths,
        ["substations", "power_substations", "substation", "power", "grid_nodes"],
        utm_crs
    )

    cell_centroids = np.column_stack([
        grid_gdf.geometry.centroid.x,
        grid_gdf.geometry.centroid.y
    ])
    max_sub_dist = float(config.get("max_substation_dist_m", 3000.0))

    if substations_gdf is None or substations_gdf.empty:
        logger.warning(
            f"No substation data found. Setting dist_nearest_substation to {max_sub_dist}m."
        )
        return pd.Series(max_sub_dist, index=grid_gdf.index, dtype=float)

    logger.info(f"Computing substation distances from {len(substations_gdf)} substations...")
    valid_subs = substations_gdf[
        substations_gdf.geometry.is_valid & ~substations_gdf.geometry.is_empty
    ]

    if valid_subs.empty:
        return pd.Series(max_sub_dist, index=grid_gdf.index, dtype=float)

    sub_pts = valid_subs.geometry.centroid
    sub_coords = np.column_stack([sub_pts.x, sub_pts.y])

    sub_tree = cKDTree(sub_coords)
    nearest_dist, _ = sub_tree.query(cell_centroids, k=1)

    return pd.Series(nearest_dist, index=grid_gdf.index, dtype=float)


# =====================================================================
# Main Feature Computation Pipeline
# =====================================================================

def compute_cell_features(
    grid_gdf: gpd.GeoDataFrame,
    processed_paths: Dict[str, Any],
    config: Dict[str, Any],
    utm_crs: Any
) -> gpd.GeoDataFrame:
    """Compute and normalize multi-modal urban features for each H3 grid cell.

    Features computed per cell:
    a. population_proxy: Count of building footprints or raster sum within cell.
    b. poi_count: Total number of points of interest within cell.
    c. poi_diversity: Number of distinct POI categories within cell.
    d. traffic_proxy: Sum of road class weights for road segments intersecting cell.
    e. dist_nearest_charger: Euclidean distance to nearest existing EV charger (meters).
    f. chargers_in_radius: Count of existing chargers within `coverage_radius_m`.
    g. dist_nearest_substation: Euclidean distance to nearest electrical substation (meters).

    In addition, min-max normalized variants [0, 1] are generated for all features.
    The enriched GeoDataFrame is saved to `data/processed/grid_features.gpkg`.

    Args:
        grid_gdf: H3 grid GeoDataFrame in UTM CRS.
        processed_paths: Mapping of layer names to file paths or GeoDataFrames.
        config: Configuration dictionary (weights, radii, thresholds).
        utm_crs: Projected coordinate reference system (UTM).

    Returns:
        Enriched GeoDataFrame with raw and normalized feature columns.
    """
    if grid_gdf is None or grid_gdf.empty:
        raise ValueError("grid_gdf cannot be empty or None")

    logger.info(f"Starting cell feature computation for {len(grid_gdf)} cells...")
    grid = grid_gdf.copy()

    # Ensure grid is in target UTM CRS
    if grid.crs != utm_crs:
        grid = grid.to_crs(utm_crs)
        if "centroid_geometry" in grid.columns:
            grid["centroid_geometry"] = gpd.GeoSeries(
                grid["centroid_geometry"],
                crs=grid_gdf.crs
            ).to_crs(utm_crs)

    # a. Population Proxy
    grid["population_proxy"] = _compute_population_proxy(grid, processed_paths, utm_crs)

    # b & c. POI Count and Diversity
    poi_count, poi_div = _compute_poi_features(grid, processed_paths, config, utm_crs)
    grid["poi_count"] = poi_count
    grid["poi_diversity"] = poi_div

    # d. Traffic Proxy
    grid["traffic_proxy"] = _compute_traffic_proxy(grid, processed_paths, config, utm_crs)

    # e & f. Charger Proximity and Density
    dist_charger, chargers_radius = _compute_charger_features(
        grid, processed_paths, config, utm_crs
    )
    grid["dist_nearest_charger"] = dist_charger
    grid["chargers_in_radius"] = chargers_radius

    # g. Substation Distance
    grid["dist_nearest_substation"] = _compute_substation_features(
        grid, processed_paths, config, utm_crs
    )

    # Min-max normalization for all features
    raw_feature_cols = [
        "population_proxy",
        "poi_count",
        "poi_diversity",
        "traffic_proxy",
        "dist_nearest_charger",
        "chargers_in_radius",
        "dist_nearest_substation",
    ]

    for col in raw_feature_cols:
        norm_col = f"{col}_norm"
        grid[norm_col] = min_max_normalize(grid[col])

    # Convenient standardized alias columns for downstream demand modeling
    grid["population_norm"] = grid["population_proxy_norm"]
    grid["poi_norm"] = grid["poi_count_norm"]
    grid["traffic_norm"] = grid["traffic_proxy_norm"]
    grid["existing_chargers_penalty_norm"] = grid["chargers_in_radius_norm"]
    grid["substation_norm"] = grid["dist_nearest_substation_norm"]

    # Export to GeoPackage
    output_target = processed_paths.get("grid_features")
    if output_target:
        output_path = Path(output_target)
    else:
        # Default path relative to project structure
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / "data" / "processed" / "grid_features.gpkg"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Prepare export DataFrame: OGC GPKG standard requires a single active geometry column
        export_gdf = grid.copy()
        if "centroid_geometry" in export_gdf.columns:
            # Convert secondary geometry column to WKT string to prevent GPKG writer conflicts
            export_gdf["centroid_geometry"] = export_gdf["centroid_geometry"].apply(
                lambda g: g.wkt if hasattr(g, "wkt") else str(g)
            )

        export_gdf.to_file(output_path, driver="GPKG", layer="grid_features")
        logger.info(f"Saved feature-enriched grid ({len(grid)} cells) to {output_path}")
    except Exception as err:
        logger.error(f"Failed to save grid features to {output_path}: {err}", exc_info=True)

    logger.info("Feature engineering complete.")
    return grid


if __name__ == "__main__":
    import yaml

    logging.info("Testing grid_features module standalone...")
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"coverage_radius_m": 1500, "h3_resolution": 8}

    # Generate a sample rectangular boundary in Austin, TX for testing
    from shapely.geometry import box
    sample_boundary = gpd.GeoDataFrame(
        {"name": ["Austin_Sample"]},
        geometry=[box(-97.77, 30.25, -97.72, 30.29)],
        crs="EPSG:4326"
    )

    utm_epsg = "EPSG:32614"  # Austin UTM Zone 14N
    resolution = int(cfg.get("h3_resolution", 8))

    try:
        h3_grid = create_h3_grid(sample_boundary, resolution=resolution, utm_crs=utm_epsg)
        logging.info(f"Generated test H3 grid with {len(h3_grid)} cells.")
        features_gdf = compute_cell_features(h3_grid, {}, cfg, utm_crs=utm_epsg)
        logging.info("Standalone test successful.")
        logging.info(f"Columns: {list(features_gdf.columns)}")
    except Exception as test_err:
        logging.error(f"Test run failed: {test_err}", exc_info=True)
