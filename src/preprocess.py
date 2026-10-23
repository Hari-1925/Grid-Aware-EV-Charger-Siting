"""Geospatial Data Preprocessing Module for EV Charger Siting.

This module cleans, standardizes, reprojects, and clips raw geospatial layers
(boundaries, road networks, points of interest, buildings, etc.) for downstream
demand modeling, clustering, and optimization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
import shapely
from shapely import make_valid
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Standardized POI category definitions
ALLOWED_POI_CATEGORIES: List[str] = [
    "office",
    "shop",
    "parking",
    "fuel",
    "transit",
    "university",
    "hospital",
    "other",
]


def clean_geometries(gdf: gpd.GeoDataFrame, layer_name: str = "") -> gpd.GeoDataFrame:
    """Filter out invalid, empty, or duplicate geometries in a GeoDataFrame.

    Uses fast vectorized operations and C-level WKB representations for
    geometry deduplication.

    Args:
        gdf: Input GeoDataFrame to clean.
        layer_name: Identifier of the layer for structured logging.

    Returns:
        Cleaned GeoDataFrame with only non-empty, valid, and unique geometries.
    """
    if gdf.empty or "geometry" not in gdf.columns:
        return gdf

    initial_count = len(gdf)

    # 1. Drop null and empty geometries
    non_empty_mask = gdf.geometry.notna() & (~gdf.geometry.is_empty)
    gdf = gdf.loc[non_empty_mask]

    # 2. Drop invalid geometries
    valid_mask = gdf.geometry.is_valid
    invalid_count = int((~valid_mask).sum())
    if invalid_count > 0:
        logger.warning(
            "Layer '%s': Dropping %d invalid geometries out of %d records",
            layer_name,
            invalid_count,
            initial_count,
        )
        gdf = gdf.loc[valid_mask]

    if gdf.empty:
        return gdf

    # 3. Deduplicate based on geometry using vectorized WKB hashing
    try:
        wkb_series = gdf.geometry.to_wkb()
        dup_mask = wkb_series.duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            logger.info(
                "Layer '%s': Dropped %d duplicate geometries",
                layer_name,
                dup_count,
            )
            gdf = gdf.loc[~dup_mask]
    except Exception as exc:
        logger.warning(
            "Layer '%s': WKB deduplication failed (%s); falling back to drop_duplicates",
            layer_name,
            exc,
        )
        gdf = gdf.drop_duplicates(subset=["geometry"], keep="first")

    return gdf


def filter_geometry_types(
    gdf: gpd.GeoDataFrame, layer_name: str
) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame to relevant geometry types based on layer category.

    - Buildings: Polygons and MultiPolygons
    - Points of Interest (POIs): Points, MultiPoints, Polygons, and MultiPolygons
    - Roads / Street networks: LineStrings and MultiLineStrings

    Args:
        gdf: Input GeoDataFrame.
        layer_name: Name of layer to infer filtering rules.

    Returns:
        GeoDataFrame filtered to valid geometry types for the layer category.
    """
    if gdf.empty or "geometry" not in gdf.columns:
        return gdf

    # Explode any GeometryCollections produced by clipping or intersections
    if (gdf.geom_type == "GeometryCollection").any():
        gdf = gdf.explode(index_parts=False)

    norm_name = layer_name.lower()

    if "building" in norm_name:
        allowed = {"Polygon", "MultiPolygon"}
    elif "poi" in norm_name:
        allowed = {"Point", "MultiPoint", "Polygon", "MultiPolygon"}
    elif any(k in norm_name for k in ["road", "highway", "network", "street"]):
        allowed = {"LineString", "MultiLineString"}
    else:
        # Default: keep all geometries for other layers (e.g. substations, boundary)
        return gdf

    type_mask = gdf.geom_type.isin(allowed)
    dropped_count = int((~type_mask).sum())
    if dropped_count > 0:
        logger.info(
            "Layer '%s': Filtered out %d geometries not matching expected types %s",
            layer_name,
            dropped_count,
            allowed,
        )
    return gdf.loc[type_mask]


def standardize_poi_categories(pois_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardize POI categories to a closed domain using vectorized conditions.

    Categories:
    - office
    - shop
    - parking
    - fuel
    - transit
    - university
    - hospital
    - other

    Args:
        pois_gdf: GeoDataFrame containing point-of-interest data.

    Returns:
        GeoDataFrame with a standardized 'category' column.
    """
    if pois_gdf.empty:
        gdf = pois_gdf.copy()
        if "category" not in gdf.columns:
            gdf["category"] = pd.Series(dtype=str)
        return gdf

    gdf = pois_gdf.copy()

    # Vectorized string series extractor
    def _extract_str_col(col_name: str) -> pd.Series:
        if col_name in gdf.columns:
            return gdf[col_name].fillna("").astype(str).str.strip().str.lower()
        return pd.Series("", index=gdf.index, dtype=str)

    cat_col = _extract_str_col("category")
    poi_type_col = _extract_str_col("poi_type")
    amenity_col = _extract_str_col("amenity")
    shop_col = _extract_str_col("shop")
    office_col = _extract_str_col("office")
    building_col = _extract_str_col("building")
    railway_col = _extract_str_col("railway")
    highway_col = _extract_str_col("highway")
    transit_col = _extract_str_col("public_transport")

    # Define vectorized boolean conditions for standard categories
    is_hospital = (
        cat_col.isin(["hospital", "clinic", "doctors"])
        | poi_type_col.isin(["hospital", "clinic", "doctors"])
        | amenity_col.isin(["hospital", "clinic", "doctors"])
        | building_col.isin(["hospital"])
    )

    is_university = (
        cat_col.isin(["university", "college"])
        | poi_type_col.isin(["university", "college"])
        | amenity_col.isin(["university", "college"])
        | building_col.isin(["university", "college"])
    )

    is_fuel = (
        cat_col.isin(["fuel", "gas_station", "charging_station"])
        | poi_type_col.isin(["fuel", "gas_station", "charging_station"])
        | amenity_col.isin(["fuel", "charging_station"])
    )

    is_parking = (
        cat_col.isin(["parking", "parking_space", "parking_entrance"])
        | poi_type_col.isin(["parking", "parking_space", "parking_entrance"])
        | amenity_col.isin(["parking", "parking_space", "parking_entrance"])
        | (
            gdf["parking"].notna() & (gdf["parking"] != "no")
            if "parking" in gdf.columns
            else False
        )
    )

    is_transit = (
        cat_col.isin(
            [
                "transit",
                "bus_station",
                "railway_station",
                "train_station",
                "subway_station",
                "station",
                "bus_stop",
                "tram_stop",
            ]
        )
        | poi_type_col.isin(
            [
                "transit",
                "bus_station",
                "railway_station",
                "train_station",
                "subway_station",
            ]
        )
        | amenity_col.isin(["bus_station"])
        | railway_col.isin(
            ["station", "halt", "subway_entrance", "tram_stop", "subway"]
        )
        | highway_col.isin(["bus_stop", "platform"])
        | transit_col.isin(["station", "stop_position", "platform", "stop_area"])
    )

    is_shop = (
        cat_col.isin(
            [
                "shop",
                "mall",
                "supermarket",
                "retail",
                "convenience",
                "department_store",
            ]
        )
        | poi_type_col.isin(["shop", "mall", "supermarket", "retail"])
        | amenity_col.isin(["marketplace"])
        | ((shop_col != "") & (shop_col != "no"))
    )

    is_office = (
        cat_col.isin(["office", "commercial"])
        | poi_type_col.isin(["office", "commercial"])
        | building_col.isin(["office", "commercial"])
        | ((office_col != "") & (office_col != "no"))
    )

    # Hierarchical priority matching
    conditions = [
        is_hospital,
        is_university,
        is_fuel,
        is_parking,
        is_transit,
        is_shop,
        is_office,
    ]
    choices = [
        "hospital",
        "university",
        "fuel",
        "parking",
        "transit",
        "shop",
        "office",
    ]

    gdf["category"] = np.select(conditions, choices, default="other")
    return gdf


def standardize_roads(roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter and standardize columns for road networks.

    Retains strictly: geometry, highway (road class), and name.

    Args:
        roads_gdf: Raw road GeoDataFrame.

    Returns:
        GeoDataFrame containing only ['geometry', 'highway', 'name'].
    """
    if roads_gdf.empty:
        gdf = roads_gdf.copy()
        for col in ["highway", "name"]:
            if col not in gdf.columns:
                gdf[col] = pd.Series(dtype=str)
        return gdf[["geometry", "highway", "name"]]

    gdf = roads_gdf.copy()

    # Ensure highway column exists and is flat string representation
    if "highway" not in gdf.columns:
        logger.warning(
            "Roads layer missing 'highway' column; setting to 'unclassified'"
        )
        gdf["highway"] = "unclassified"
    else:
        # Handle cases where highway contains list values (common in OSMnx)
        if gdf["highway"].dtype == object:
            gdf["highway"] = gdf["highway"].apply(
                lambda val: val[0]
                if isinstance(val, (list, tuple)) and len(val) > 0
                else (str(val) if pd.notna(val) else "unclassified")
            )
        gdf["highway"] = gdf["highway"].astype(str)

    # Ensure name column exists and is flat string representation
    if "name" not in gdf.columns:
        gdf["name"] = ""
    else:
        if gdf["name"].dtype == object:
            gdf["name"] = gdf["name"].apply(
                lambda val: str(val[0])
                if isinstance(val, (list, tuple)) and len(val) > 0
                else (str(val) if pd.notna(val) else "")
            )
        gdf["name"] = gdf["name"].fillna("").astype(str)

    # Select only required columns
    return gdf[["geometry", "highway", "name"]]


def sanitize_for_gpkg(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure non-geometry columns are compatible with GeoPackage serialization.

    Converts lists, tuples, or dictionaries in object columns to strings to prevent
    GDAL GPKG driver export errors.

    Args:
        gdf: GeoDataFrame to sanitize.

    Returns:
        Sanitized GeoDataFrame.
    """
    if gdf.empty:
        return gdf

    gdf = gdf.copy()
    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == object:
            sample = gdf[col].dropna()
            if not sample.empty and any(
                isinstance(v, (list, dict, tuple)) for v in sample.iloc[:50]
            ):
                gdf[col] = gdf[col].apply(
                    lambda v: ", ".join(map(str, v))
                    if isinstance(v, (list, tuple))
                    else (str(v) if isinstance(v, dict) else v)
                )
    return gdf


def preprocess_data(
    raw_paths: Dict[str, Union[str, Path]],
    config: Optional[Dict[str, Any]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Clean and preprocess raw geospatial datasets for EV charger siting.

    Pipeline operations:
    1. Loads raw GeoPackage files from raw_paths.
    2. Identifies and cleans the boundary polygon, resolving MultiPolygons via unary_union.
    3. Estimates optimal projected UTM CRS from the boundary and reprojects all layers.
    4. Removes invalid, empty, and duplicate geometries.
    5. Filters mixed geometry types to appropriate layer types (polygons for buildings,
       points/polygons for POIs, lines for roads).
    6. Clips all spatial layers to the boundary geometry.
    7. Standardizes POI categories and road attribute columns.
    8. Writes cleaned datasets to base_dir / data / processed / as GPKG files.

    Args:
        raw_paths: Dictionary mapping layer names to their raw file paths.
        config: Configuration dictionary (e.g. from config.yaml).
        base_dir: Project root directory path. Defaults to repository root.
        use_cache: If True, reuse existing processed files when valid.

    Returns:
        Dictionary mapping layer names to processed GPKG file paths, plus 'utm_crs'.
    """
    config = config or {}

    # Resolve base_dir and processed directory
    if base_dir is None:
        resolved_base_dir = Path(__file__).resolve().parent.parent
    else:
        resolved_base_dir = Path(base_dir)

    if resolved_base_dir.name == "data":
        processed_dir = resolved_base_dir / "processed"
    else:
        processed_dir = resolved_base_dir / "data" / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processed data destination directory: %s", processed_dir)

    # 1. Identify boundary layer key
    boundary_key = next(
        (k for k in raw_paths if "boundary" in k.lower()),
        "boundary" if "boundary" in raw_paths else None,
    )

    if not boundary_key or boundary_key not in raw_paths:
        err_msg = (
            "A 'boundary' layer is required in raw_paths to determine UTM CRS "
            "and clip layers. Available keys: " + str(list(raw_paths.keys()))
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    # 2. Check cache if enabled
    all_cached = True
    if use_cache or config.get("use_cache", False):
        expected_paths = {
            k: processed_dir / f"{k}.gpkg" for k in raw_paths.keys()
        }
        for k, p in expected_paths.items():
            if not p.exists() or p.stat().st_size == 0:
                all_cached = False
                break
        if all_cached:
            logger.info("All processed datasets found in cache; loading boundary CRS.")
            cached_boundary = gpd.read_file(expected_paths[boundary_key])
            cached_crs = cached_boundary.crs
            return {
                **expected_paths,
                "processed_paths": expected_paths,
                "utm_crs": cached_crs,
                "crs": cached_crs,
            }

    # 3. Load and prepare boundary layer
    boundary_path = Path(raw_paths[boundary_key])
    if not boundary_path.exists():
        err_msg = f"Boundary raw file does not exist at {boundary_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    try:
        boundary_gdf = gpd.read_file(boundary_path)
    except Exception as exc:
        logger.error("Failed to load boundary file %s: %s", boundary_path, exc)
        raise

    if boundary_gdf.empty:
        err_msg = f"Boundary layer at {boundary_path} is empty."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Clean boundary geometries
    boundary_gdf = clean_geometries(boundary_gdf, layer_name=boundary_key)

    # Ensure boundary has CRS
    if boundary_gdf.crs is None:
        logger.warning("Boundary layer has no CRS defined; defaulting to EPSG:4326")
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")

    # Estimate UTM CRS from boundary
    try:
        utm_crs = boundary_gdf.estimate_utm_crs()
        logger.info("Estimated UTM CRS from boundary: %s", utm_crs)
    except Exception as exc:
        logger.warning(
            "estimate_utm_crs() failed (%s); using existing boundary CRS: %s",
            exc,
            boundary_gdf.crs,
        )
        utm_crs = boundary_gdf.crs

    # Reproject boundary to UTM CRS
    if boundary_gdf.crs != utm_crs:
        boundary_gdf = boundary_gdf.to_crs(utm_crs)

    # Edge Case: MultiPolygon boundary -> use unary_union
    unified_boundary = boundary_gdf.geometry.unary_union
    if unified_boundary.geom_type == "GeometryCollection":
        poly_parts = [
            g
            for g in unified_boundary.geoms
            if g.geom_type in ("Polygon", "MultiPolygon")
        ]
        unified_boundary = unary_union(poly_parts)

    if not unified_boundary.is_valid:
        logger.warning("Repaired invalid unified boundary geometry using make_valid")
        unified_boundary = make_valid(unified_boundary)

    # Create unified boundary GeoDataFrame
    boundary_gdf = gpd.GeoDataFrame(
        {"geometry": [unified_boundary]}, crs=utm_crs
    )

    processed_paths: Dict[str, Path] = {}

    # Save cleaned boundary
    boundary_out = processed_dir / f"{boundary_key}.gpkg"
    try:
        boundary_gdf.to_file(boundary_out, driver="GPKG")
        processed_paths[boundary_key] = boundary_out
        logger.info(
            "Saved cleaned boundary to %s (geom_type: %s)",
            boundary_out,
            unified_boundary.geom_type,
        )
    except Exception as exc:
        logger.error("Failed to save boundary GPKG: %s", exc)
        raise

    # 4. Process all remaining layers
    for layer_name, raw_path in raw_paths.items():
        if layer_name == boundary_key:
            continue

        raw_p = Path(raw_path)
        logger.info("Processing layer '%s' from %s", layer_name, raw_p)

        if not raw_p.exists():
            logger.warning(
                "Raw file for layer '%s' does not exist (%s). Skipping.",
                layer_name,
                raw_p,
            )
            continue

        try:
            gdf = gpd.read_file(raw_p)
        except Exception as exc:
            logger.error(
                "Failed to read layer '%s' from %s: %s", layer_name, raw_p, exc
            )
            continue

        # Edge case: Empty dataframe -> log warning and continue
        if gdf.empty:
            logger.warning(
                "Raw layer '%s' is empty. Writing empty dataset and continuing.",
                layer_name,
            )
            out_p = processed_dir / f"{layer_name}.gpkg"
            try:
                gdf = gdf.set_crs(utm_crs, allow_override=True)
                gdf.to_file(out_p, driver="GPKG")
                processed_paths[layer_name] = out_p
            except Exception as exc:
                logger.warning(
                    "Could not persist empty layer '%s': %s", layer_name, exc
                )
            continue

        # Remove duplicates and invalid geometries
        gdf = clean_geometries(gdf, layer_name=layer_name)
        if gdf.empty:
            logger.warning(
                "Layer '%s' has no valid geometries remaining after cleaning. Continuing.",
                layer_name,
            )
            continue

        # Filter mixed geometry types
        gdf = filter_geometry_types(gdf, layer_name=layer_name)
        if gdf.empty:
            logger.warning(
                "Layer '%s' is empty after filtering geometry types. Continuing.",
                layer_name,
            )
            continue

        # Reproject to UTM CRS
        if gdf.crs is None:
            logger.warning(
                "Layer '%s' missing CRS; assuming EPSG:4326 before reprojection.",
                layer_name,
            )
            gdf = gdf.set_crs("EPSG:4326")

        if gdf.crs != utm_crs:
            gdf = gdf.to_crs(utm_crs)

        # Clip layer to boundary
        try:
            gdf = gpd.clip(gdf, boundary_gdf)
        except Exception as exc:
            logger.error(
                "Error clipping layer '%s' to boundary: %s", layer_name, exc
            )
            continue

        # Clean any empty or invalid geometry remnants resulting from clip
        gdf = gdf.loc[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
        gdf = clean_geometries(gdf, layer_name=layer_name)
        gdf = filter_geometry_types(gdf, layer_name=layer_name)

        if gdf.empty:
            logger.warning(
                "Layer '%s' is empty after boundary clipping. Continuing.",
                layer_name,
            )
            continue

        # Layer-specific standardization
        norm_layer = layer_name.lower()
        if "poi" in norm_layer:
            gdf = standardize_poi_categories(gdf)
        elif any(
            k in norm_layer for k in ["road", "highway", "network", "street"]
        ):
            gdf = standardize_roads(gdf)

        # Sanitize non-geometry columns for GPKG export
        gdf = sanitize_for_gpkg(gdf)

        # Save processed layer to GPKG
        out_path = processed_dir / f"{layer_name}.gpkg"
        try:
            gdf.to_file(out_path, driver="GPKG")
            processed_paths[layer_name] = out_path
            logger.info(
                "Successfully saved processed layer '%s' to %s (%d records)",
                layer_name,
                out_path,
                len(gdf),
            )
        except Exception as exc:
            logger.error(
                "Failed to write processed GPKG for layer '%s' to %s: %s",
                layer_name,
                out_path,
                exc,
            )

    return {
        **processed_paths,
        "processed_paths": processed_paths,
        "utm_crs": utm_crs,
        "crs": utm_crs,
    }


if __name__ == "__main__":
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    root_dir = Path(__file__).resolve().parent.parent
    config_file = root_dir / "config.yaml"
    cfg = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            logger.info("Loaded configuration from %s", config_file)
        except Exception as err:
            logger.warning("Could not read %s: %s", config_file, err)

    raw_dir = root_dir / "data" / "raw"
    sample_raw = {
        p.stem: p for p in raw_dir.glob("*.gpkg")
    } if raw_dir.exists() else {}

    if sample_raw:
        logger.info("Found raw GPKG files: %s", list(sample_raw.keys()))
        results = preprocess_data(sample_raw, cfg, root_dir)
        logger.info("Preprocessing complete. Output paths: %s", results)
    else:
        logger.info("No raw GPKG files found in %s to process directly.", raw_dir)
