"""Optimal EV Charger Siting - Optimization Module.

Implements the Maximal Covering Location Problem (MCLP) using PuLP and integer
programming, with geometric feasibility filtering and a greedy fallback heuristic.
All spatial distance computations are performed using scipy.spatial.cKDTree in
projected metric coordinate reference systems (CRS).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pulp
import scipy.sparse
from scipy.spatial import cKDTree
import yaml

logger = logging.getLogger(__name__)


def _parse_config(config: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """Parse configuration from a dictionary or a YAML file path.

    Args:
        config: Configuration dictionary or path to a YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration key-value mapping.

    Raises:
        FileNotFoundError: If a configuration file path does not exist.
        TypeError: If the config argument is neither a dict nor a valid path.
    """
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    elif isinstance(config, dict):
        return config
    else:
        raise TypeError(f"Expected config to be a dict, str, or Path, got {type(config).__name__}")


def _get_metric_crs(gdf: gpd.GeoDataFrame) -> Any:
    """Determine an appropriate projected metric CRS for distance computations.

    If the GeoDataFrame is already projected, its CRS is preserved. If it is in a
    geographic CRS (e.g., EPSG:4326) or unassigned, it estimates an appropriate UTM
    zone projection.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        Projected metric CRS (pyproj.CRS or compatible string/int).
    """
    if gdf.crs is not None and not gdf.crs.is_geographic:
        return gdf.crs

    try:
        temp_gdf = gdf if gdf.crs is not None else gdf.set_crs(epsg=4326)
        estimated_crs = temp_gdf.estimate_utm_crs()
        if estimated_crs is not None:
            return estimated_crs
    except Exception as exc:
        logger.warning("Could not estimate UTM CRS (%s); falling back to EPSG:3857.", exc)

    return "EPSG:3857"


def _extract_coords(gdf: gpd.GeoDataFrame, target_crs: Any) -> np.ndarray:
    """Reproject geometries to the target metric CRS and extract 2D coordinates.

    Uses geometry centroids to support Point, Polygon, and MultiPolygon geometries
    (e.g., H3 hexagonal cells or parcel polygons).

    Args:
        gdf: GeoDataFrame containing spatial geometries.
        target_crs: Target projected coordinate reference system in meters.

    Returns:
        np.ndarray: 2D array of shape (N, 2) containing metric (X, Y) coordinates.
    """
    if len(gdf) == 0:
        return np.empty((0, 2), dtype=float)

    try:
        temp_gdf = gdf if gdf.crs is not None else gdf.set_crs(epsg=4326)
        proj_gdf = temp_gdf.to_crs(target_crs) if temp_gdf.crs != target_crs else temp_gdf
        centroids = proj_gdf.geometry.centroid
        return np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    except Exception as exc:
        logger.error("Failed to extract projected coordinates: %s", exc)
        raise


def _get_demand_weights(demand_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Extract demand weights from a demand GeoDataFrame.

    Searches across common column names ('unmet_demand', 'demand', 'demand_score',
    'score', 'weight', 'total_demand'). If none is found, falls back to a uniform
    weight of 1.0 per cell.

    Args:
        demand_gdf: GeoDataFrame containing demand cells or centroids.

    Returns:
        np.ndarray: 1D array of non-negative demand values with length len(demand_gdf).
    """
    candidate_cols = [
        "unmet_demand",
        "demand",
        "demand_score",
        "score",
        "weight",
        "total_demand",
    ]
    for col in candidate_cols:
        if col in demand_gdf.columns:
            weights = demand_gdf[col].to_numpy(dtype=float)
            weights = np.nan_to_num(weights, nan=0.0)
            return np.maximum(weights, 0.0)

    logger.warning(
        "No demand column found in demand_gdf (checked: %s). Defaulting to uniform demand = 1.0.",
        candidate_cols,
    )
    return np.ones(len(demand_gdf), dtype=float)


def feasibility_filter(
    candidates_gdf: gpd.GeoDataFrame,
    chargers_gdf: Optional[gpd.GeoDataFrame],
    substations_gdf: Optional[gpd.GeoDataFrame],
    config: Union[Dict[str, Any], str, Path],
) -> gpd.GeoDataFrame:
    """Filter candidate sites based on existing charger spacing and substation proximity.

    Removes candidates within min_spacing_m of existing chargers and candidates beyond
    max_substation_dist_m of any electrical substation. If fewer than n_new_stations
    candidates remain after filtering, max_substation_dist_m is relaxed by 50% increments
    until enough candidates remain or all charger-spaced candidates are included.

    Args:
        candidates_gdf: GeoDataFrame of candidate charger locations.
        chargers_gdf: Optional GeoDataFrame of existing charger locations.
        substations_gdf: Optional GeoDataFrame of electrical substations.
        config: Configuration dictionary or path containing optimization parameters.

    Returns:
        gpd.GeoDataFrame: Filtered candidate locations satisfying spacing and proximity rules.
    """
    cfg = _parse_config(config)
    n_new_stations = int(cfg.get("n_new_stations", 20))
    min_spacing_m = float(cfg.get("min_spacing_m", 800.0))
    max_substation_dist_m = float(cfg.get("max_substation_dist_m", 3000.0))

    if candidates_gdf is None or len(candidates_gdf) == 0:
        logger.warning("Empty candidates_gdf provided to feasibility_filter.")
        return candidates_gdf.copy() if candidates_gdf is not None else gpd.GeoDataFrame()

    n_candidates = len(candidates_gdf)
    metric_crs = _get_metric_crs(candidates_gdf)
    cand_coords = _extract_coords(candidates_gdf, metric_crs)

    # 1. Spacing filter against existing chargers
    if chargers_gdf is not None and len(chargers_gdf) > 0:
        charger_coords = _extract_coords(chargers_gdf, metric_crs)
        charger_tree = cKDTree(charger_coords)
        dists_to_chargers, _ = charger_tree.query(cand_coords, k=1)
        charger_spacing_mask = dists_to_chargers >= min_spacing_m
        removed_count = n_candidates - int(np.count_nonzero(charger_spacing_mask))
        logger.info(
            "Charger spacing filter: removed %d candidates within %.1fm of existing chargers (%d remaining).",
            removed_count,
            min_spacing_m,
            int(np.count_nonzero(charger_spacing_mask)),
        )
    else:
        logger.info("No existing chargers provided; skipping charger spacing filter.")
        charger_spacing_mask = np.ones(n_candidates, dtype=bool)

    available_after_chargers = int(np.count_nonzero(charger_spacing_mask))
    if available_after_chargers == 0:
        logger.warning("No candidates remain after existing charger spacing filter.")
        return candidates_gdf.iloc[0:0].copy()

    # 2. Substation proximity filter with adaptive relaxation
    if substations_gdf is not None and len(substations_gdf) > 0:
        sub_coords = _extract_coords(substations_gdf, metric_crs)
        sub_tree = cKDTree(sub_coords)
        dists_to_substations, _ = sub_tree.query(cand_coords, k=1)

        current_max_dist = max_substation_dist_m
        target_count = min(n_new_stations, available_after_chargers)
        relaxation_iterations = 0
        max_relax_iterations = 25  # Avoid infinite loop

        while True:
            combined_mask = charger_spacing_mask & (dists_to_substations <= current_max_dist)
            num_remaining = int(np.count_nonzero(combined_mask))

            if num_remaining >= target_count or relaxation_iterations >= max_relax_iterations:
                break

            old_dist = current_max_dist
            current_max_dist *= 1.5
            relaxation_iterations += 1
            logger.info(
                "Feasibility filter: only %d candidates remain within %.1fm (< target %d). "
                "Relaxing max_substation_dist_m by 50%% from %.1fm to %.1fm (iteration %d).",
                num_remaining,
                old_dist,
                target_count,
                old_dist,
                current_max_dist,
                relaxation_iterations,
            )

        final_mask = charger_spacing_mask & (dists_to_substations <= current_max_dist)
        if relaxation_iterations > 0:
            logger.info(
                "Substation distance filter relaxed %d time(s). Final threshold: %.1fm, retained candidates: %d.",
                relaxation_iterations,
                current_max_dist,
                int(np.count_nonzero(final_mask)),
            )
        else:
            logger.info(
                "Substation distance filter applied at %.1fm: %d candidates retained.",
                current_max_dist,
                int(np.count_nonzero(final_mask)),
            )
    else:
        logger.warning("No substations provided; skipping substation distance filter.")
        final_mask = charger_spacing_mask

    filtered_gdf = candidates_gdf[final_mask].copy()
    logger.info(
        "Feasibility filtering complete: %d of %d original candidates retained.",
        len(filtered_gdf),
        n_candidates,
    )
    return filtered_gdf


def build_coverage_matrix(
    candidates_gdf: gpd.GeoDataFrame,
    demand_gdf: gpd.GeoDataFrame,
    coverage_radius_m: float,
) -> Tuple[scipy.sparse.csr_matrix, Dict[str, Any]]:
    """Build a sparse binary coverage matrix between candidates and demand cells.

    A demand cell i is covered by candidate j if the Euclidean distance between them
    in metric projection is <= coverage_radius_m. Uses scipy.spatial.cKDTree.query_ball_point
    for efficient neighbor lookup.

    Args:
        candidates_gdf: GeoDataFrame of candidate charger locations (columns: j).
        demand_gdf: GeoDataFrame of demand cells or centroids (rows: i).
        coverage_radius_m: Service coverage radius in meters.

    Returns:
        Tuple of:
            - coverage_matrix: scipy.sparse.csr_matrix of shape (len(demand_gdf), len(candidates_gdf))
              where coverage[i, j] = 1 if candidate j covers demand cell i, else 0.
            - index_mappings: Dictionary containing row/column index mappings and original IDs.
    """
    n_demand = len(demand_gdf)
    n_candidates = len(candidates_gdf)

    index_mappings = {
        "candidate_indices": candidates_gdf.index.tolist(),
        "demand_indices": demand_gdf.index.tolist(),
        "candidate_to_col": {idx: col for col, idx in enumerate(candidates_gdf.index)},
        "demand_to_row": {idx: row for row, idx in enumerate(demand_gdf.index)},
        "col_to_candidate": dict(enumerate(candidates_gdf.index)),
        "row_to_demand": dict(enumerate(demand_gdf.index)),
    }

    if n_demand == 0 or n_candidates == 0:
        logger.warning(
            "Empty GeoDataFrame in build_coverage_matrix (demand cells: %d, candidates: %d).",
            n_demand,
            n_candidates,
        )
        empty_mat = scipy.sparse.csr_matrix((n_demand, n_candidates), dtype=np.int8)
        return empty_mat, index_mappings

    metric_crs = _get_metric_crs(candidates_gdf)
    cand_coords = _extract_coords(candidates_gdf, metric_crs)
    demand_coords = _extract_coords(demand_gdf, metric_crs)

    cand_tree = cKDTree(cand_coords)
    neighbors = cand_tree.query_ball_point(demand_coords, r=float(coverage_radius_m))

    row_counts = np.fromiter((len(c) for c in neighbors), dtype=np.int64, count=n_demand)
    total_nnz = int(row_counts.sum())

    if total_nnz > 0:
        rows = np.repeat(np.arange(n_demand, dtype=np.int32), row_counts)
        cols = np.concatenate([np.asarray(c, dtype=np.int32) for c in neighbors if len(c) > 0])
        data = np.ones(total_nnz, dtype=np.int8)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.int8)

    coverage_matrix = scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_demand, n_candidates),
        dtype=np.int8,
    )

    logger.info(
        "Coverage matrix constructed: shape (%d demand cells, %d candidates), non-zero entries: %d.",
        n_demand,
        n_candidates,
        total_nnz,
    )

    return coverage_matrix, index_mappings


def greedy_fallback(
    candidates_gdf: gpd.GeoDataFrame,
    demand_gdf: gpd.GeoDataFrame,
    config: Union[Dict[str, Any], str, Path],
) -> gpd.GeoDataFrame:
    """Greedy heuristic fallback for EV charger site selection.

    Iteratively selects the candidate site covering the highest amount of currently
    uncovered demand, while strictly enforcing the minimum pairwise spacing constraint
    between selected stations.

    Args:
        candidates_gdf: GeoDataFrame of candidate charger locations.
        demand_gdf: GeoDataFrame of demand cells or centroids.
        config: Configuration dictionary or path containing optimization parameters.

    Returns:
        gpd.GeoDataFrame: Selected charger locations with 'demand_covered' per site.
    """
    cfg = _parse_config(config)
    n_new_stations = int(cfg.get("n_new_stations", 20))
    coverage_radius_m = float(cfg.get("coverage_radius_m", 1500.0))
    min_spacing_m = float(cfg.get("min_spacing_m", 800.0))

    if candidates_gdf is None or len(candidates_gdf) == 0:
        logger.warning("Empty candidates GeoDataFrame passed to greedy_fallback.")
        empty_gdf = candidates_gdf.copy() if candidates_gdf is not None else gpd.GeoDataFrame()
        empty_gdf["demand_covered"] = pd.Series(dtype=float)
        return empty_gdf

    n_candidates = len(candidates_gdf)
    n_demand = len(demand_gdf)

    if n_demand == 0 or n_new_stations <= 0:
        logger.warning("No demand cells or zero station budget in greedy_fallback.")
        budget = min(n_new_stations, n_candidates)
        res = candidates_gdf.iloc[:budget].copy()
        res["demand_covered"] = 0.0
        return res

    metric_crs = _get_metric_crs(candidates_gdf)
    cand_coords = _extract_coords(candidates_gdf, metric_crs)
    cand_tree = cKDTree(cand_coords)

    coverage_matrix, _ = build_coverage_matrix(candidates_gdf, demand_gdf, coverage_radius_m)
    coverage_csc = coverage_matrix.tocsc()

    demand_weights = _get_demand_weights(demand_gdf)
    uncovered_demand = demand_weights.copy()

    # Precalculate total demand covered per candidate site
    site_total_demand = np.asarray(coverage_matrix.T.dot(demand_weights)).flatten()

    available_candidates: Set[int] = set(range(n_candidates))
    selected_indices: List[int] = []

    for step in range(n_new_stations):
        if not available_candidates:
            logger.info("No more available candidates satisfying spacing constraints. Halting greedy selection.")
            break

        # Vectorized marginal demand computation across all candidates
        marginal_demands = np.asarray(coverage_matrix.T.dot(uncovered_demand)).flatten()

        # Find available candidate with maximum marginal covered demand
        best_candidate = max(available_candidates, key=lambda j: marginal_demands[j])
        best_gain = marginal_demands[best_candidate]

        if best_gain <= 0.0:
            logger.info(
                "All reachable demand has been covered. Halting greedy selection at %d stations (target %d).",
                len(selected_indices),
                n_new_stations,
            )
            break

        selected_indices.append(best_candidate)
        available_candidates.remove(best_candidate)

        # Mark demand cells covered by best_candidate as satisfied
        covered_cell_indices = coverage_csc.getcol(best_candidate).indices
        uncovered_demand[covered_cell_indices] = 0.0

        # Enforce pairwise spacing: remove conflicting candidates within min_spacing_m
        conflict_indices = cand_tree.query_ball_point(cand_coords[best_candidate], r=min_spacing_m)
        for conflict in conflict_indices:
            available_candidates.discard(conflict)

    selected_gdf = candidates_gdf.iloc[selected_indices].copy()
    selected_gdf["demand_covered"] = [
        round(float(site_total_demand[j]), 2) for j in selected_indices
    ]
    selected_gdf = selected_gdf.sort_values(by="demand_covered", ascending=False).reset_index(drop=True)

    logger.info(
        "Greedy fallback completed: selected %d stations out of target %d.",
        len(selected_gdf),
        n_new_stations,
    )
    return selected_gdf


def solve_mclp(
    candidates_gdf: gpd.GeoDataFrame,
    demand_gdf: gpd.GeoDataFrame,
    config: Union[Dict[str, Any], str, Path],
) -> gpd.GeoDataFrame:
    """Solve the Maximal Covering Location Problem (MCLP) using PuLP.

    Formulates and solves the integer programming problem:
        Maximize sum(unmet_demand_i * y_i)
        Subject to:
            y_i <= sum(x_j for j covering i)      forall i in demand
            sum(x_j) <= n_new_stations
            x_j1 + x_j2 <= 1                      forall pairs (j1, j2) with dist < min_spacing_m
            x_j, y_i in {0, 1}

    Uses the CBC solver with a configurable time limit. If solver execution fails
    or cannot find a valid solution, automatically invokes greedy_fallback.

    Args:
        candidates_gdf: GeoDataFrame of candidate charger locations.
        demand_gdf: GeoDataFrame of demand cells or centroids.
        config: Configuration dictionary or path containing optimization parameters.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of selected sites with 'demand_covered' per site.
    """
    cfg = _parse_config(config)
    n_new_stations = int(cfg.get("n_new_stations", 20))
    coverage_radius_m = float(cfg.get("coverage_radius_m", 1500.0))
    min_spacing_m = float(cfg.get("min_spacing_m", 800.0))
    solver_time_limit_s = float(cfg.get("solver_time_limit_s", 60.0))

    if candidates_gdf is None or len(candidates_gdf) == 0:
        logger.warning("Empty candidates GeoDataFrame passed to solve_mclp.")
        empty_gdf = candidates_gdf.copy() if candidates_gdf is not None else gpd.GeoDataFrame()
        empty_gdf["demand_covered"] = pd.Series(dtype=float)
        return empty_gdf

    n_candidates = len(candidates_gdf)
    n_demand = len(demand_gdf)

    if n_demand == 0 or n_new_stations <= 0:
        logger.warning("No demand cells or zero station budget in solve_mclp.")
        return greedy_fallback(candidates_gdf, demand_gdf, config)

    metric_crs = _get_metric_crs(candidates_gdf)
    cand_coords = _extract_coords(candidates_gdf, metric_crs)

    try:
        # Build binary coverage matrix and demand weights
        coverage_matrix, _ = build_coverage_matrix(candidates_gdf, demand_gdf, coverage_radius_m)
        demand_weights = _get_demand_weights(demand_gdf)
        site_total_demand = np.asarray(coverage_matrix.T.dot(demand_weights)).flatten()

        # Identify pairwise spacing conflicts
        cand_tree = cKDTree(cand_coords)
        conflict_pairs = cand_tree.query_pairs(r=min_spacing_m)

        # Set up PuLP Optimization Model
        prob = pulp.LpProblem("EV_Charger_MCLP", pulp.LpMaximize)

        # Decision variables
        x = {j: pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(n_candidates)}
        y = {i: pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary) for i in range(n_demand)}

        # Objective: Maximize total covered demand
        prob += (
            pulp.lpSum([float(demand_weights[i]) * y[i] for i in range(n_demand)]),
            "Total_Covered_Demand",
        )

        # Constraint 1: Station budget limit
        budget = min(n_new_stations, n_candidates)
        prob += pulp.lpSum(list(x.values())) <= budget, "Station_Budget_Constraint"

        # Constraint 2: Demand coverage linkages y_i <= sum(x_j for j covering i)
        for i in range(n_demand):
            covering_candidates = coverage_matrix.indices[
                coverage_matrix.indptr[i] : coverage_matrix.indptr[i + 1]
            ]
            if len(covering_candidates) > 0:
                prob += y[i] <= pulp.lpSum([x[j] for j in covering_candidates]), f"Cover_Demand_{i}"
            else:
                prob += y[i] == 0, f"Uncoverable_Demand_{i}"

        # Constraint 3: Pairwise minimum spacing constraints
        for pair_idx, (j1, j2) in enumerate(conflict_pairs):
            prob += x[j1] + x[j2] <= 1, f"Spacing_Conflict_{pair_idx}_{j1}_{j2}"

        # Solve using CBC solver
        solver = pulp.PULP_CBC_CMD(
            timeLimit=solver_time_limit_s,
            msg=False,
            gapRel=0.01,
        )
        solver_status = prob.solve(solver)
        status_name = pulp.LpStatus.get(solver_status, "Unknown")
        logger.info("PuLP MCLP solver finished with status: %s", status_name)

        # Extract selected candidates
        selected_indices = [
            j
            for j in range(n_candidates)
            if x[j].varValue is not None and x[j].varValue > 0.5
        ]

        if len(selected_indices) == 0:
            logger.warning("PuLP solver yielded no selected sites. Invoking greedy fallback.")
            return greedy_fallback(candidates_gdf, demand_gdf, config)

        selected_gdf = candidates_gdf.iloc[selected_indices].copy()
        selected_gdf["demand_covered"] = [
            round(float(site_total_demand[j]), 2) for j in selected_indices
        ]
        selected_gdf = selected_gdf.sort_values(by="demand_covered", ascending=False).reset_index(drop=True)

        covered_cells_count = sum(
            1 for i in range(n_demand) if y[i].varValue is not None and y[i].varValue > 0.5
        )
        total_covered_demand = sum(
            float(demand_weights[i])
            for i in range(n_demand)
            if y[i].varValue is not None and y[i].varValue > 0.5
        )
        logger.info(
            "MCLP solved successfully: %d sites selected, covering %d demand cells (total demand covered: %.2f).",
            len(selected_gdf),
            covered_cells_count,
            total_covered_demand,
        )

        return selected_gdf

    except Exception as exc:
        logger.error("Exception encountered in solve_mclp (%s). Falling back to greedy heuristic.", exc)
        return greedy_fallback(candidates_gdf, demand_gdf, config)
