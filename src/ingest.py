"""Data ingestion module for EV Charger Siting.

Downloads and caches geographic, network, POI, infrastructure, and demographic
data using OSMnx and WorldPop for the specified city.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
import yaml

# Configure logger
logger = logging.getLogger(__name__)

# Configure OSMnx settings for caching and network requests
if hasattr(ox, "settings"):
    for timeout_attr in ("requests_timeout", "timeout"):
        if hasattr(ox.settings, timeout_attr):
            setattr(ox.settings, timeout_attr, 300)
    if hasattr(ox.settings, "use_cache"):
        ox.settings.use_cache = True
    if hasattr(ox.settings, "log_console"):
        ox.settings.log_console = False

# Default tag definitions for OSM features
DEFAULT_POI_TAGS: dict[str, Any] = {
    "amenity": [
        "parking",
        "fuel",
        "bus_station",
        "university",
        "hospital",
    ],
    "office": True,
    "shop": True,
    "railway": ["station"],
    "building": [
        "university",
        "hospital",
        "office",
        "commercial",
        "retail",
    ],
}

DEFAULT_CHARGER_TAGS: dict[str, Any] = {
    "amenity": ["charging_station"],
}

DEFAULT_SUBSTATION_TAGS: dict[str, Any] = {
    "power": ["substation"],
}

DEFAULT_BUILDING_TAGS: dict[str, Any] = {
    "building": True,
}


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. Defaults to
            'config.yaml' in the project root if None.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If parsing the configuration file fails.
    """
    if config_path is None:
        target_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        target_path = Path(config_path)

    if not target_path.exists():
        msg = f"Configuration file not found at: {target_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            logger.info("Successfully loaded configuration from %s", target_path)
            return cfg or {}
    except Exception as exc:
        logger.error("Failed to read or parse configuration file %s: %s", target_path, exc)
        raise


def _sanitize_for_geopackage(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Prepare a GeoDataFrame for storage in GeoPackage (.gpkg) format.

    Handles MultiIndex, nested list/dict column values, duplicate column
    names, and null/empty geometries to ensure safe serialization.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: Cleaned GeoDataFrame ready for GPKG export.
    """
    if gdf is None or gdf.empty:
        if gdf is None:
            return gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                crs="EPSG:4326",
            )
        gdf_clean = gdf.copy()
        if gdf_clean.crs is None:
            gdf_clean.set_crs("EPSG:4326", inplace=True)
        return gdf_clean

    clean = gdf.copy()

    # Reset MultiIndex or named index so index levels become regular columns
    if isinstance(clean.index, pd.MultiIndex) or clean.index.name is not None:
        clean = clean.reset_index()

    # Drop duplicate columns if any resulted from reset_index or joins
    clean = clean.loc[:, ~clean.columns.duplicated()]

    # Filter out empty or null geometries
    if "geometry" in clean.columns:
        valid_mask = clean.geometry.notna() & ~clean.geometry.is_empty
        clean = clean.loc[valid_mask]

    # Ensure valid coordinate reference system
    if clean.crs is None:
        clean = clean.set_crs("EPSG:4326")

    geom_col = clean.geometry.name if hasattr(clean, "geometry") else "geometry"

    # Convert non-scalar values (lists, tuples, dicts) to strings for SQLite/GPKG
    for col in clean.columns:
        if col == geom_col:
            continue
        if clean[col].dtype == "object":
            clean[col] = clean[col].apply(
                lambda val: ", ".join(str(item) for item in val)
                if isinstance(val, (list, tuple, set))
                else json.dumps(val)
                if isinstance(val, dict)
                else val
            )
            # If column still contains mixed types (e.g. ints and strings), normalize
            valid_vals = clean[col].dropna()
            if not valid_vals.empty and valid_vals.map(type).nunique() > 1:
                clean[col] = clean[col].apply(lambda x: str(x) if pd.notna(x) else None)

    return clean


def _save_geodataframe(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    layer_name: str,
) -> Path:
    """Save a GeoDataFrame to a GeoPackage (.gpkg) file safely.

    Args:
        gdf: GeoDataFrame to save.
        output_path: Target .gpkg file path.
        layer_name: Layer name inside the GeoPackage.

    Returns:
        Path: Path to saved GeoPackage file.

    Raises:
        RuntimeError: If saving fails.
    """
    try:
        sanitized = _sanitize_for_geopackage(gdf)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sanitized.to_file(output_path, layer=layer_name, driver="GPKG")
        logger.info(
            "Saved layer '%s' with %d features to %s",
            layer_name,
            len(sanitized),
            output_path,
        )
        return output_path
    except Exception as exc:
        logger.error("Failed to save GeoDataFrame to %s: %s", output_path, exc)
        raise RuntimeError(f"Failed writing GPKG to {output_path}: {exc}") from exc


def download_boundary(city: str, output_path: Path) -> Path:
    """Download and cache city boundary polygon using OSMnx geocode_to_gdf.

    Args:
        city: City name string for geocoding (e.g., 'Austin, Texas, USA').
        output_path: Destination path for boundary.gpkg.

    Returns:
        Path: Path to the cached boundary.gpkg.

    Raises:
        RuntimeError: If geocoding or saving fails.
    """
    if output_path.exists():
        logger.info("Cached city boundary already exists at %s. Skipping download.", output_path)
        return output_path

    logger.info("Downloading city boundary for '%s'...", city)
    try:
        boundary_gdf = ox.geocode_to_gdf(city)
        if boundary_gdf.empty:
            raise ValueError(f"No boundary found for query: '{city}'")
        return _save_geodataframe(boundary_gdf, output_path, layer_name="boundary")
    except Exception as exc:
        logger.error("Failed downloading city boundary for '%s': %s", city, exc)
        raise RuntimeError(f"Error downloading boundary for '{city}': {exc}") from exc


def download_roads(city: str, output_path: Path) -> Path:
    """Download and cache drivable road network edges using OSMnx.

    Args:
        city: City name string for query.
        output_path: Destination path for roads.gpkg.

    Returns:
        Path: Path to the cached roads.gpkg.

    Raises:
        RuntimeError: If graph download or processing fails.
    """
    if output_path.exists():
        logger.info("Cached road network already exists at %s. Skipping download.", output_path)
        return output_path

    logger.info("Downloading drivable road network for '%s'...", city)
    try:
        graph = ox.graph_from_place(city, network_type="drive")
        _, edges_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        if edges_gdf.empty:
            raise ValueError(f"Empty road network returned for '{city}'")
        logger.info("Extracted %d road segments for '%s'", len(edges_gdf), city)
        return _save_geodataframe(edges_gdf, output_path, layer_name="roads")
    except Exception as exc:
        logger.error("Failed downloading road network for '%s': %s", city, exc)
        raise RuntimeError(f"Error downloading road network for '{city}': {exc}") from exc


def download_chargers(city: str, output_path: Path) -> Path:
    """Download and cache existing EV charging stations using OSMnx.

    Args:
        city: City name string.
        output_path: Destination path for chargers.gpkg.

    Returns:
        Path: Path to the cached chargers.gpkg.

    Raises:
        RuntimeError: If download fails unexpectedly.
    """
    if output_path.exists():
        logger.info("Cached chargers file already exists at %s. Skipping download.", output_path)
        return output_path

    logger.info("Downloading charging stations for '%s'...", city)
    try:
        gdf = ox.features_from_place(city, tags=DEFAULT_CHARGER_TAGS)
        logger.info("Downloaded %d charging stations for '%s'", len(gdf), city)
        return _save_geodataframe(gdf, output_path, layer_name="chargers")
    except Exception as exc:
        msg = str(exc).lower()
        if "no data elements" in msg or "insufficient" in msg or "empty" in msg:
            logger.warning("No charging stations found in OSM for '%s': %s", city, exc)
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            return _save_geodataframe(empty_gdf, output_path, layer_name="chargers")
        logger.error("Failed downloading charging stations for '%s': %s", city, exc)
        raise RuntimeError(f"Error downloading charging stations for '{city}': {exc}") from exc


def download_substations(city: str, output_path: Path) -> Path:
    """Download and cache power substations using OSMnx.

    Args:
        city: City name string.
        output_path: Destination path for substations.gpkg.

    Returns:
        Path: Path to the cached substations.gpkg.

    Raises:
        RuntimeError: If download fails unexpectedly.
    """
    if output_path.exists():
        logger.info("Cached substations file already exists at %s. Skipping download.", output_path)
        return output_path

    logger.info("Downloading power substations for '%s'...", city)
    try:
        gdf = ox.features_from_place(city, tags=DEFAULT_SUBSTATION_TAGS)
        logger.info("Downloaded %d power substations for '%s'", len(gdf), city)
        return _save_geodataframe(gdf, output_path, layer_name="substations")
    except Exception as exc:
        msg = str(exc).lower()
        if "no data elements" in msg or "insufficient" in msg or "empty" in msg:
            logger.warning("No substations found in OSM for '%s': %s", city, exc)
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            return _save_geodataframe(empty_gdf, output_path, layer_name="substations")
        logger.error("Failed downloading power substations for '%s': %s", city, exc)
        raise RuntimeError(f"Error downloading substations for '{city}': {exc}") from exc


def download_pois(
    city: str,
    output_path: Path,
    tags: Optional[dict[str, Any]] = None,
) -> Path:
    """Download and cache Points of Interest (POIs) using OSMnx.

    Args:
        city: City name string.
        output_path: Destination path for pois.gpkg.
        tags: Optional dictionary of OSM tags. Uses DEFAULT_POI_TAGS if None.

    Returns:
        Path: Path to the cached pois.gpkg.

    Raises:
        RuntimeError: If download fails unexpectedly.
    """
    if output_path.exists():
        logger.info("Cached POIs file already exists at %s. Skipping download.", output_path)
        return output_path

    query_tags = tags or DEFAULT_POI_TAGS
    logger.info("Downloading POIs for '%s' with tags: %s", city, query_tags)
    try:
        gdf = ox.features_from_place(city, tags=query_tags)
        logger.info("Downloaded %d POIs for '%s'", len(gdf), city)
        return _save_geodataframe(gdf, output_path, layer_name="pois")
    except Exception as exc:
        msg = str(exc).lower()
        if "no data elements" in msg or "insufficient" in msg or "empty" in msg:
            logger.warning("No POIs found in OSM for '%s': %s", city, exc)
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            return _save_geodataframe(empty_gdf, output_path, layer_name="pois")
        logger.error("Failed downloading POIs for '%s': %s", city, exc)
        raise RuntimeError(f"Error downloading POIs for '{city}': {exc}") from exc


def download_buildings(city: str, output_path: Path) -> Path:
    """Download and cache building footprints using OSMnx.

    Building footprints serve as a proxy for local population and urban activity.

    Args:
        city: City name string.
        output_path: Destination path for buildings.gpkg.

    Returns:
        Path: Path to the cached buildings.gpkg.

    Raises:
        RuntimeError: If download fails unexpectedly.
    """
    if output_path.exists():
        logger.info("Cached buildings file already exists at %s. Skipping download.", output_path)
        return output_path

    logger.info("Downloading building footprints for '%s'...", city)
    try:
        gdf = ox.features_from_place(city, tags=DEFAULT_BUILDING_TAGS)
        logger.info("Downloaded %d building footprints for '%s'", len(gdf), city)
        return _save_geodataframe(gdf, output_path, layer_name="buildings")
    except Exception as exc:
        msg = str(exc).lower()
        if "no data elements" in msg or "insufficient" in msg or "empty" in msg:
            logger.warning("No buildings found in OSM for '%s': %s", city, exc)
            empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            return _save_geodataframe(empty_gdf, output_path, layer_name="buildings")
        logger.error("Failed downloading building footprints for '%s': %s", city, exc)
        raise RuntimeError(f"Error downloading buildings for '{city}': {exc}") from exc


def _infer_iso3_country_code(city: str, config: Optional[dict[str, Any]] = None) -> str:
    """Infer the ISO3 country code from config or city string.

    Args:
        city: City name string.
        config: Optional configuration dictionary.

    Returns:
        str: ISO3 country code (e.g., 'USA').
    """
    if config and config.get("iso3"):
        return str(config["iso3"]).upper().strip()

    city_upper = city.upper() if city else ""
    if any(k in city_upper for k in ("USA", "UNITED STATES", "TEXAS", "CALIFORNIA", "FLORIDA", "NEW YORK")):
        return "USA"
    if any(k in city_upper for k in ("CANADA", "CAN")):
        return "CAN"
    if any(k in city_upper for k in ("UNITED KINGDOM", "UK", "GBR", "ENGLAND", "SCOTLAND")):
        return "GBR"
    if any(k in city_upper for k in ("GERMANY", "DEU", "DEUTSCHLAND")):
        return "DEU"
    if any(k in city_upper for k in ("INDIA", "IND")):
        return "IND"

    parts = [p.strip() for p in city.split(",") if p.strip()]
    if parts and len(parts[-1]) == 3 and parts[-1].isalpha():
        return parts[-1].upper()

    return "USA"


def fetch_population_data(
    config: dict[str, Any],
    raw_dir: Path,
    timeout_s: int = 30,
) -> Path:
    """Attempt to download WorldPop raster data with fallback to building proxy.

    Tries the WorldPop REST API with the specified timeout. If the request fails,
    times out, or errors, sets population_source to 'building_proxy', logs the
    fallback, and writes the source flag to data/raw/population_source.txt.

    Args:
        config: Configuration dictionary with city or ISO3 info.
        raw_dir: Path to directory for raw data storage.
        timeout_s: HTTP request timeout in seconds (default 30).

    Returns:
        Path: Path to worldpop.tif if successful, or population_source.txt on fallback.
    """
    flag_path = raw_dir / "population_source.txt"
    raster_path = raw_dir / "worldpop.tif"

    # Check if population source flag or raster is already cached
    if flag_path.exists():
        try:
            cached_source = flag_path.read_text(encoding="utf-8").strip()
            if cached_source == "worldpop" and raster_path.exists():
                logger.info("Found cached WorldPop raster at %s", raster_path)
                return raster_path
            if cached_source == "building_proxy":
                logger.info(
                    "Cached population source indicates 'building_proxy' at %s. Skipping download.",
                    flag_path,
                )
                return flag_path
        except Exception as exc:
            logger.warning("Could not read cached population flag at %s: %s", flag_path, exc)

    city = config.get("city", "Austin, Texas, USA")
    iso3 = _infer_iso3_country_code(city, config)
    logger.info("Attempting WorldPop population download for ISO3 '%s' (timeout=%ds)...", iso3, timeout_s)

    try:
        api_url = f"https://hub.worldpop.org/rest/data/pop/wpgp?iso3={iso3}"
        logger.info("Querying WorldPop REST API: %s", api_url)
        response = requests.get(api_url, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()

        items = data.get("data", [])
        if not items:
            raise ValueError(f"WorldPop API returned no datasets for ISO3 '{iso3}'")

        target_download_url: Optional[str] = None
        for item in items:
            files = item.get("files", [])
            for f in files:
                if isinstance(f, str) and f.lower().endswith(".tif"):
                    target_download_url = f
                    break
            if target_download_url:
                break

        if not target_download_url:
            raise ValueError("No TIF download links found in WorldPop API response")

        logger.info("Downloading WorldPop raster from %s...", target_download_url)
        with requests.get(target_download_url, stream=True, timeout=timeout_s) as download_resp:
            download_resp.raise_for_status()
            with open(raster_path, "wb") as f_out:
                for chunk in download_resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f_out.write(chunk)

        flag_path.write_text("worldpop", encoding="utf-8")
        logger.info("WorldPop raster successfully downloaded to %s", raster_path)
        return raster_path

    except Exception as exc:
        logger.warning(
            "WorldPop download failed (%s). Falling back to building_proxy.",
            exc,
        )
        flag_path.write_text("building_proxy", encoding="utf-8")
        logger.info("Saved population fallback source 'building_proxy' to %s", flag_path)
        return flag_path


def ingest_data(config: dict[str, Any], base_dir: Path) -> dict[str, Path]:
    """Ingest and cache all raw data required for the EV charger siting project.

    Downloads city boundary, road network edges, existing charging stations,
    electrical substations, POIs, building footprints, and population data.
    All outputs are cached in <base_dir>/data/raw/.

    Args:
        config: Configuration dictionary loaded from config.yaml.
        base_dir: Project root directory path.

    Returns:
        dict[str, Path]: Dictionary containing paths to all downloaded raw files:
            - 'boundary': Path to boundary.gpkg
            - 'roads': Path to roads.gpkg
            - 'chargers': Path to chargers.gpkg
            - 'substations': Path to substations.gpkg
            - 'pois': Path to pois.gpkg
            - 'buildings': Path to buildings.gpkg
            - 'population': Path to worldpop.tif or population_source.txt
            - 'population_source': Path to population_source.txt
    """
    base_dir = Path(base_dir).resolve()
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    city = config.get("city", "Austin, Texas, USA")
    logger.info("Starting data ingestion for city: '%s'", city)
    logger.info("Target raw directory: %s", raw_dir)

    # 1. City boundary
    boundary_path = download_boundary(city, raw_dir / "boundary.gpkg")

    # 2. Road network edges
    roads_path = download_roads(city, raw_dir / "roads.gpkg")

    # 3. Existing charging stations
    chargers_path = download_chargers(city, raw_dir / "chargers.gpkg")

    # 4. Power substations
    substations_path = download_substations(city, raw_dir / "substations.gpkg")

    # 5. Points of Interest (POIs)
    pois_path = download_pois(city, raw_dir / "pois.gpkg")

    # 6. Building footprints (for population proxy)
    buildings_path = download_buildings(city, raw_dir / "buildings.gpkg")

    # 7. Population data (WorldPop with building proxy fallback)
    population_path = fetch_population_data(config, raw_dir, timeout_s=30)
    flag_path = raw_dir / "population_source.txt"

    results = {
        "boundary": boundary_path,
        "roads": roads_path,
        "chargers": chargers_path,
        "substations": substations_path,
        "pois": pois_path,
        "buildings": buildings_path,
        "population": population_path,
        "population_source": flag_path,
    }

    logger.info("Data ingestion completed successfully. Summary of artifacts:")
    for key, path in results.items():
        logger.info("  %s -> %s", key, path)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    project_root = Path(__file__).resolve().parent.parent
    configuration = load_config(project_root / "config.yaml")
    ingest_data(configuration, project_root)
