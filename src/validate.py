"""
Validation Module for EV Charger Siting Optimization.

Provides model validation, baseline comparisons, and sensitivity analysis:
1. Holdout Validation: Hides 20% of existing chargers, recomputes supply gap
   and demand, solves MCLP, and evaluates spatial recovery rate.
2. Baseline Comparison: Benchmarks our method against Random and Population-Only
   baselines, saving an illustrative bar chart.
3. Sensitivity Analysis: Evaluates solution stability across perturbations in
   feature weights (+/-20%), coverage radius (+/-500m), and station count (+/-5).
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Attempt to import optimization components from src.optimize
try:
    from src.optimize import feasibility_filter, solve_mclp
except ImportError:
    feasibility_filter = None
    solve_mclp = None


# ---------------------------------------------------------------------------
# Spatial & Normalization Utilities
# ---------------------------------------------------------------------------

def _to_metric_crs(gdf: gpd.GeoDataFrame, target_crs: Optional[Any] = None) -> gpd.GeoDataFrame:
    """
    Ensure GeoDataFrame is in a projected metric CRS for distance calculations.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame.
    target_crs : Any, optional
        Target metric CRS. If None, estimates UTM or falls back to EPSG:3857.

    Returns
    -------
    gpd.GeoDataFrame
        Reprojected GeoDataFrame.
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
        except Exception:
            return gdf.to_crs(epsg=3857)

    return gdf


def _extract_xy_coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Extract (N, 2) array of coordinates from point or polygon centroids."""
    if gdf is None or gdf.empty:
        return np.empty((0, 2), dtype=float)
    centroids = gdf.geometry.centroid
    return np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])


def _min_max_scale(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to [0, 1]."""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    s_min = s.min()
    s_max = s.max()
    if s_max > s_min:
        return (s - s_min) / (s_max - s_min)
    return pd.Series(0.0, index=series.index)


def _recompute_grid_supply_and_demand(
    grid_gdf: gpd.GeoDataFrame,
    chargers_gdf: Optional[gpd.GeoDataFrame],
    config: Dict[str, Any],
) -> gpd.GeoDataFrame:
    """
    Recalculate supply gap, demand, and unmet demand on grid cells.

    Recalculates:
    - chargers_in_radius: count of chargers within coverage_radius_m
    - dist_nearest_charger: distance to nearest charger
    - supply_gap: normalized undersupply metric
    - demand: composite score from population, POIs, and traffic
    - unmet_demand: demand scaled by supply gap and existing charger penalty

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Input grid cells.
    chargers_gdf : Optional[gpd.GeoDataFrame]
        Active chargers to compute supply from.
    config : Dict[str, Any]
        Configuration with weights and radii.

    Returns
    -------
    gpd.GeoDataFrame
        Updated copy of grid_gdf with recalculated supply and demand columns.
    """
    grid = grid_gdf.copy()
    radius_m = float(config.get("coverage_radius_m", 1500.0))

    grid_proj = _to_metric_crs(grid)
    grid_coords = _extract_xy_coords(grid_proj)
    n_cells = len(grid_coords)

    if chargers_gdf is not None and len(chargers_gdf) > 0:
        chargers_proj = _to_metric_crs(chargers_gdf, target_crs=grid_proj.crs)
        charger_coords = _extract_xy_coords(chargers_proj)
        tree = cKDTree(charger_coords)

        in_radius_counts = np.array(
            [len(idx) for idx in tree.query_ball_point(grid_coords, r=radius_m)],
            dtype=float,
        )
        nearest_dists, _ = tree.query(grid_coords, k=1)
        grid["chargers_in_radius"] = in_radius_counts
        grid["dist_nearest_charger"] = nearest_dists
    else:
        grid["chargers_in_radius"] = 0.0
        grid["dist_nearest_charger"] = 50000.0

    # Renormalize supply features
    in_radius_norm = _min_max_scale(grid["chargers_in_radius"])
    dist_norm = _min_max_scale(grid["dist_nearest_charger"])

    # Supply gap: higher when distant and fewer chargers nearby
    supply_gap = 0.5 * (1.0 - in_radius_norm) + 0.5 * dist_norm
    grid["supply_gap"] = supply_gap

    # Feature weights
    weights = config.get("feature_weights", {})
    w_pop = float(weights.get("population", 0.35))
    w_poi = float(weights.get("poi", 0.25))
    w_traf = float(weights.get("traffic", 0.25))
    w_pen = float(weights.get("existing_chargers_penalty", 0.15))

    # Detect population column
    pop_val = pd.Series(0.5, index=grid.index)
    for col in ["population_proxy", "population_norm", "population", "pop"]:
        if col in grid.columns:
            pop_val = _min_max_scale(grid[col])
            break

    # Detect POI column
    poi_val = pd.Series(0.5, index=grid.index)
    for col in ["poi_norm", "poi_count", "poi", "poi_score"]:
        if col in grid.columns:
            poi_val = _min_max_scale(grid[col])
            break

    # Detect traffic column
    traf_val = pd.Series(0.5, index=grid.index)
    for col in ["traffic_norm", "traffic_score", "traffic_volume", "traffic"]:
        if col in grid.columns:
            traf_val = _min_max_scale(grid[col])
            break

    w_sum = w_pop + w_poi + w_traf
    base_demand = (w_pop * pop_val + w_poi * poi_val + w_traf * traf_val) / (w_sum if w_sum > 0 else 1.0)
    grid["demand"] = base_demand

    # Unmet demand combines base demand, supply gap, and existing charger penalty
    penalty_factor = np.clip(1.0 - (w_pen * in_radius_norm), 0.05, 1.0)
    unmet_demand = base_demand * supply_gap * penalty_factor
    grid["unmet_demand"] = np.clip(unmet_demand, 0.0, None)

    return grid


def _resolve_candidates(
    grid_gdf: gpd.GeoDataFrame,
    candidates_gdf: Optional[gpd.GeoDataFrame],
    base_dir: Union[str, Path],
) -> gpd.GeoDataFrame:
    """Retrieve or generate candidate sites from inputs, disk, or grid centroids."""
    if candidates_gdf is not None and not candidates_gdf.empty:
        return candidates_gdf.copy()

    # Attempt to load candidate geojson from outputs or data directory
    base_path = Path(base_dir)
    for potential_file in [
        base_path / "outputs" / "candidate_sites.geojson",
        base_path / "data" / "candidate_sites.geojson",
        base_path / "data" / "candidates.geojson",
    ]:
        if potential_file.exists():
            try:
                loaded = gpd.read_file(potential_file)
                if not loaded.empty:
                    logger.debug("Loaded candidates from %s", potential_file)
                    return loaded
            except Exception as e:
                logger.warning("Could not read candidate file %s: %s", potential_file, e)

    # Fallback: create candidates from top 30% grid cell centroids
    grid_copy = grid_gdf.copy()
    sort_col = "unmet_demand" if "unmet_demand" in grid_copy.columns else "demand"
    if sort_col in grid_copy.columns:
        n_top = max(20, int(len(grid_copy) * 0.30))
        top_grid = grid_copy.nlargest(n_top, sort_col)
    else:
        top_grid = grid_copy

    cand_gdf = gpd.GeoDataFrame(
        top_grid.drop(columns=["geometry"], errors="ignore"),
        geometry=top_grid.geometry.centroid,
        crs=grid_gdf.crs,
    )
    return cand_gdf


def _internal_mclp_solve(
    grid_gdf: gpd.GeoDataFrame,
    candidates_gdf: gpd.GeoDataFrame,
    config: Dict[str, Any],
) -> gpd.GeoDataFrame:
    """
    Fallback Maximum Covering Location Problem (MCLP) solver.

    Uses PuLP mixed-integer linear programming if available,
    otherwise uses a deterministic greedy coverage algorithm.
    """
    n_sites = int(config.get("n_new_stations", 20))
    radius_m = float(config.get("coverage_radius_m", 1500.0))
    time_limit = int(config.get("solver_time_limit_s", 60))

    if len(candidates_gdf) <= n_sites:
        return candidates_gdf.copy()

    # Project to metric CRS
    grid_proj = _to_metric_crs(grid_gdf)
    cands_proj = _to_metric_crs(candidates_gdf, target_crs=grid_proj.crs)

    grid_coords = _extract_xy_coords(grid_proj)
    cand_coords = _extract_xy_coords(cands_proj)

    demand_col = "unmet_demand" if "unmet_demand" in grid_proj.columns else "demand"
    demand_weights = (
        grid_proj[demand_col].to_numpy(dtype=float)
        if demand_col in grid_proj.columns
        else np.ones(len(grid_coords), dtype=float)
    )

    # Coverage matrix via cKDTree
    cand_tree = cKDTree(cand_coords)
    cell_neighbors = cand_tree.query_ball_point(grid_coords, r=radius_m)

    # Invert to cand -> list of covered grid cell indices
    cand_coverage = {j: [] for j in range(len(cand_coords))}
    for i, nbrs in enumerate(cell_neighbors):
        for j in nbrs:
            cand_coverage[j].append(i)

    # Try PuLP formulation
    try:
        import pulp

        model = pulp.LpProblem("MCLP_Validation", pulp.LpMaximize)

        # Variables
        y = [pulp.LpVariable(f"y_{j}", cat=pulp.LpBinary) for j in range(len(cand_coords))]
        x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(len(grid_coords))]

        # Objective
        model += pulp.lpSum(demand_weights[i] * x[i] for i in range(len(grid_coords)))

        # Constraints
        model += pulp.lpSum(y) <= n_sites, "Max_Stations"

        for i, cand_indices in enumerate(cell_neighbors):
            if cand_indices:
                model += x[i] <= pulp.lpSum(y[j] for j in cand_indices)
            else:
                model += x[i] == 0

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        status = model.solve(solver)

        if pulp.LpStatus[status] == "Optimal" or pulp.LpStatus[status] == "Feasible":
            selected_indices = [j for j in range(len(cand_coords)) if pulp.value(y[j]) > 0.5]
            if len(selected_indices) > 0:
                selected_cands = candidates_gdf.iloc[selected_indices].copy()
                selected_cands["demand_covered"] = [
                    float(demand_weights[cand_coverage[j]].sum()) for j in selected_indices
                ]
                return selected_cands.sort_values(by="demand_covered", ascending=False)
    except Exception as e:
        logger.debug("PuLP solver unavailable or failed (%s); using greedy fallback", e)

    # Greedy fallback
    selected_cands_idx: List[int] = []
    uncovered_cells = set(range(len(grid_coords)))

    for _ in range(n_sites):
        best_cand = None
        best_gain = -1.0
        for j in range(len(cand_coords)):
            if j in selected_cands_idx:
                continue
            cells_covered = set(cand_coverage[j]) & uncovered_cells
            gain = demand_weights[list(cells_covered)].sum() if cells_covered else 0.0
            if gain > best_gain:
                best_gain = gain
                best_cand = j

        if best_cand is None or best_gain <= 0:
            remaining = [j for j in range(len(cand_coords)) if j not in selected_cands_idx]
            if remaining:
                selected_cands_idx.append(remaining[0])
            break

        selected_cands_idx.append(best_cand)
        uncovered_cells -= set(cand_coverage[best_cand])

    selected_cands = candidates_gdf.iloc[selected_cands_idx].copy()
    selected_cands["demand_covered"] = [
        float(demand_weights[cand_coverage[j]].sum()) for j in selected_cands_idx
    ]
    return selected_cands.sort_values(by="demand_covered", ascending=False)


def _execute_solver(
    solve_func: Optional[Callable],
    grid_gdf: gpd.GeoDataFrame,
    candidates_gdf: gpd.GeoDataFrame,
    config: Dict[str, Any],
) -> gpd.GeoDataFrame:
    """Execute provided solver or fall back to solve_mclp / internal solver."""
    target_func = solve_func or solve_mclp
    if target_func is not None:
        try:
            sig = inspect.signature(target_func)
            param_count = len(sig.parameters)
            if param_count == 2:
                res = target_func(grid_gdf, candidates_gdf)
            elif param_count >= 3:
                res = target_func(grid_gdf, candidates_gdf, config)
            else:
                res = target_func(grid_gdf, candidates_gdf, config)

            if isinstance(res, tuple):
                res = res[0]
            if isinstance(res, gpd.GeoDataFrame):
                return res
        except Exception as e:
            logger.warning("External solve_func failed (%s), running internal solver", e)

    return _internal_mclp_solve(grid_gdf, candidates_gdf, config)


# ---------------------------------------------------------------------------
# Core Module Functions
# ---------------------------------------------------------------------------

def holdout_validation(
    grid_gdf: gpd.GeoDataFrame,
    chargers_gdf: gpd.GeoDataFrame,
    config: Dict[str, Any],
    base_dir: Union[str, Path],
    solve_func: Optional[Callable] = None,
    candidates_gdf: Optional[gpd.GeoDataFrame] = None,
    substations_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Dict[str, Union[float, int]]:
    """
    Perform spatial holdout validation by hiding 20% of existing chargers.

    Workflow:
    1. Randomly hide 20% of existing chargers (holdout set).
    2. Recalculate grid supply gap, demand, and unmet demand using remaining chargers.
    3. Run feasibility filtering and MCLP solver with remaining chargers.
    4. Check what % of hidden chargers fall within 500m of any recommended site.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Grid cells with demand and spatial coordinates.
    chargers_gdf : gpd.GeoDataFrame
        Existing EV chargers to sample from.
    config : Dict[str, Any]
        Configuration options (holdout fraction, proximity threshold, etc.).
    base_dir : Union[str, Path]
        Project root path.
    solve_func : Optional[Callable]
        Solver function. If None, uses src.optimize.solve_mclp or fallback.
    candidates_gdf : Optional[gpd.GeoDataFrame], optional
        Candidate siting locations.
    substations_gdf : Optional[gpd.GeoDataFrame], optional
        Electrical substations for feasibility filtering.

    Returns
    -------
    Dict[str, Union[float, int]]
        Dictionary containing 'holdout_recovery_rate', 'n_hidden', and 'n_recovered'.
    """
    logger.info("Starting holdout validation...")
    try:
        val_cfg = config.get("validation", {})
        holdout_fraction = float(val_cfg.get("holdout_fraction", 0.20))
        proximity_m = float(val_cfg.get("holdout_proximity_m", 500.0))

        if chargers_gdf is None or len(chargers_gdf) < 2:
            logger.warning("Fewer than 2 chargers provided; cannot perform holdout validation.")
            return {"holdout_recovery_rate": 0.0, "n_hidden": 0, "n_recovered": 0}

        # 1. Randomly hide 20% of existing chargers
        n_total = len(chargers_gdf)
        n_hidden = max(1, int(round(n_total * holdout_fraction)))

        np.random.seed(42)
        hidden_indices = np.random.choice(chargers_gdf.index, size=n_hidden, replace=False)
        hidden_chargers = chargers_gdf.loc[hidden_indices].copy()
        remaining_chargers = chargers_gdf.drop(index=hidden_indices).copy()

        logger.info(
            "Holdout split: %d total chargers -> %d remaining, %d hidden (%.1f%%)",
            n_total,
            len(remaining_chargers),
            n_hidden,
            (n_hidden / n_total) * 100.0,
        )

        # 2. Recompute supply gap and unmet demand without hidden chargers
        grid_recomputed = _recompute_grid_supply_and_demand(grid_gdf, remaining_chargers, config)

        # 3. Resolve candidates and apply feasibility filter
        resolved_cands = _resolve_candidates(grid_recomputed, candidates_gdf, base_dir)

        filter_func = feasibility_filter
        if filter_func is not None and substations_gdf is not None and not substations_gdf.empty:
            try:
                filtered_cands = filter_func(resolved_cands, substations_gdf, config)
                if filtered_cands is not None and not filtered_cands.empty:
                    resolved_cands = filtered_cands
            except Exception as e:
                logger.warning("Feasibility filtering in holdout failed (%s); proceeding with unfiltered", e)

        # 4. Run optimization pipeline
        recommended_sites = _execute_solver(solve_func, grid_recomputed, resolved_cands, config)

        if recommended_sites is None or recommended_sites.empty:
            logger.error("Solver returned empty recommendations during holdout validation.")
            return {"holdout_recovery_rate": 0.0, "n_hidden": n_hidden, "n_recovered": 0}

        # 5. Measure proximity of hidden chargers to recommended sites
        rec_proj = _to_metric_crs(recommended_sites)
        hidden_proj = _to_metric_crs(hidden_chargers, target_crs=rec_proj.crs)

        rec_coords = _extract_xy_coords(rec_proj)
        hidden_coords = _extract_xy_coords(hidden_proj)

        rec_tree = cKDTree(rec_coords)
        nearest_dists, _ = rec_tree.query(hidden_coords, k=1)

        recovered_mask = nearest_dists <= proximity_m
        n_recovered = int(np.sum(recovered_mask))
        recovery_rate = float(n_recovered / n_hidden) if n_hidden > 0 else 0.0

        results = {
            "holdout_recovery_rate": round(recovery_rate, 4),
            "n_hidden": n_hidden,
            "n_recovered": n_recovered,
        }

        logger.info(
            "Holdout validation complete: Recovery Rate = %.2f%% (%d/%d within %.0fm)",
            recovery_rate * 100.0,
            n_recovered,
            n_hidden,
            proximity_m,
        )
        return results

    except Exception as e:
        logger.error("Holdout validation failed: %s", str(e), exc_info=True)
        return {"holdout_recovery_rate": 0.0, "n_hidden": 0, "n_recovered": 0}


def baseline_comparison(
    grid_gdf: gpd.GeoDataFrame,
    candidates_gdf: gpd.GeoDataFrame,
    config: Dict[str, Any],
    base_dir: Union[str, Path],
) -> Dict[str, Union[float, int, str]]:
    """
    Benchmark optimization method against Random and Population-Only baselines.

    Compares:
    - Random baseline: Average unmet demand coverage across 50 stochastic placements.
    - Population-only baseline: Selects N sites with highest population proxy.
    - Our method: MCLP optimization solution.

    Saves comparison bar chart to outputs/baseline_comparison.png.

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Grid cells with unmet demand.
    candidates_gdf : gpd.GeoDataFrame
        Candidate locations for station placement.
    config : Dict[str, Any]
        Configuration options (n_new_stations, coverage_radius_m, etc.).
    base_dir : Union[str, Path]
        Project root path.

    Returns
    -------
    Dict[str, Union[float, int, str]]
        Dictionary with coverage percentages and execution parameters.
    """
    logger.info("Starting baseline comparison benchmark...")
    try:
        val_cfg = config.get("validation", {})
        n_runs = int(val_cfg.get("random_baseline_runs", 50))
        n_sites = int(config.get("n_new_stations", 20))
        radius_m = float(config.get("coverage_radius_m", 1500.0))

        # Project grid and candidates
        grid_proj = _to_metric_crs(grid_gdf)
        cands_proj = _to_metric_crs(candidates_gdf, target_crs=grid_proj.crs)

        grid_coords = _extract_xy_coords(grid_proj)
        cand_coords = _extract_xy_coords(cands_proj)
        n_cands = len(cands_proj)
        n_select = min(n_sites, n_cands)

        # Unmet demand array
        demand_col = "unmet_demand" if "unmet_demand" in grid_proj.columns else "demand"
        if demand_col in grid_proj.columns:
            demand_arr = grid_proj[demand_col].to_numpy(dtype=float)
        else:
            demand_arr = np.ones(len(grid_coords), dtype=float)

        total_demand = demand_arr.sum()

        def _measure_coverage(site_coords_subset: np.ndarray) -> float:
            if len(site_coords_subset) == 0:
                return 0.0
            site_tree = cKDTree(site_coords_subset)
            covered_indices = site_tree.query_ball_point(grid_coords, r=radius_m)
            covered_mask = np.array([len(idx) > 0 for idx in covered_indices], dtype=bool)
            if total_demand > 0:
                return float((demand_arr[covered_mask].sum() / total_demand) * 100.0)
            return float(np.mean(covered_mask) * 100.0)

        # ---------------------------------------------------------
        # 1. Random Baseline (Average over n_runs)
        # ---------------------------------------------------------
        np.random.seed(42)
        random_coverages: List[float] = []
        for _ in range(n_runs):
            rand_indices = np.random.choice(n_cands, size=n_select, replace=False)
            cov = _measure_coverage(cand_coords[rand_indices])
            random_coverages.append(cov)

        random_mean = float(np.mean(random_coverages))
        random_std = float(np.std(random_coverages))

        # ---------------------------------------------------------
        # 2. Population-Only Baseline
        # ---------------------------------------------------------
        pop_col = None
        for col in ["population_proxy", "population", "pop_proxy", "pop"]:
            if col in cands_proj.columns:
                pop_col = col
                break

        if pop_col is not None:
            top_pop_indices = cands_proj[pop_col].to_numpy().argsort()[::-1][:n_select]
        else:
            # Spatial join or nearest grid cell population proxy
            grid_pop_col = None
            for col in ["population_proxy", "population", "pop"]:
                if col in grid_proj.columns:
                    grid_pop_col = col
                    break

            if grid_pop_col is not None:
                grid_tree = cKDTree(grid_coords)
                _, nearest_grid_idx = grid_tree.query(cand_coords, k=1)
                cand_pop_scores = grid_proj[grid_pop_col].iloc[nearest_grid_idx].to_numpy()
                top_pop_indices = cand_pop_scores.argsort()[::-1][:n_select]
            else:
                top_pop_indices = np.arange(n_select)

        pop_coverage = _measure_coverage(cand_coords[top_pop_indices])

        # ---------------------------------------------------------
        # 3. Our Method (MCLP Solution)
        # ---------------------------------------------------------
        our_sites = _execute_solver(None, grid_gdf, candidates_gdf, config)
        our_proj = _to_metric_crs(our_sites, target_crs=grid_proj.crs)
        our_coords = _extract_xy_coords(our_proj)
        our_coverage = _measure_coverage(our_coords)

        results = {
            "random_baseline": round(random_mean, 2),
            "random_baseline_std": round(random_std, 2),
            "population_only": round(pop_coverage, 2),
            "our_method": round(our_coverage, 2),
            "n_stations": n_select,
            "coverage_radius_m": radius_m,
            "metric": "percent_unmet_demand_covered",
        }

        # ---------------------------------------------------------
        # 4. Save Bar Chart Comparison
        # ---------------------------------------------------------
        base_path = Path(base_dir)
        outputs_dir = base_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        chart_path = outputs_dir / "baseline_comparison.png"

        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        methods = ["Random Baseline\n(50-Run Avg)", "Population-Only\nHeuristic", "Our Method\n(Grid-Aware MCLP)"]
        values = [random_mean, pop_coverage, our_coverage]
        colors = ["#95a5a6", "#2980b9", "#27ae60"]

        bars = ax.bar(
            methods,
            values,
            color=colors,
            width=0.55,
            edgecolor="#2c3e50",
            linewidth=1.2,
            yerr=[random_std, 0, 0],
            capsize=5,
            error_kw={"elinewidth": 1.2, "ecolor": "#34495e"},
        )

        # Annotate bar values
        for bar, val in zip(bars, values):
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 1.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                color="#2c3e50",
            )

        ax.set_ylabel("Unmet Demand Covered (%)", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_title(
            f"Coverage Benchmark Comparison (N = {n_select} Chargers, R = {int(radius_m)}m)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        results["chart_path"] = str(chart_path)
        logger.info(
            "Baseline comparison complete: Random=%.1f%%, Pop=%.1f%%, Our Method=%.1f%%. Saved chart to %s",
            random_mean,
            pop_coverage,
            our_coverage,
            chart_path,
        )
        return results

    except Exception as e:
        logger.error("Baseline comparison failed: %s", str(e), exc_info=True)
        return {
            "random_baseline": 0.0,
            "population_only": 0.0,
            "our_method": 0.0,
            "error": str(e),
        }


def sensitivity_analysis(
    grid_gdf: gpd.GeoDataFrame,
    candidates_gdf: gpd.GeoDataFrame,
    chargers_gdf: gpd.GeoDataFrame,
    substations_gdf: Optional[gpd.GeoDataFrame],
    config: Dict[str, Any],
    base_dir: Union[str, Path],
    recommended_sites: Optional[gpd.GeoDataFrame] = None,
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis on weights, radius, and station count.

    Perturbations evaluated:
    - Feature weights: +/- 20% for each weight category.
    - Coverage radius: +/- 500m.
    - Station count (N): +/- 5 stations.

    Stability metric:
    Percentage of original optimal sites retained in each perturbed solution
    (spatial proximity <= 300m or identical index).

    Parameters
    ----------
    grid_gdf : gpd.GeoDataFrame
        Grid cells with spatial demand features.
    candidates_gdf : gpd.GeoDataFrame
        Candidate locations.
    chargers_gdf : gpd.GeoDataFrame
        Existing EV chargers.
    substations_gdf : Optional[gpd.GeoDataFrame]
        Electrical substations for feasibility filtering.
    config : Dict[str, Any]
        Base configuration dictionary.
    base_dir : Union[str, Path]
        Project root path.
    recommended_sites : Optional[gpd.GeoDataFrame], optional
        Baseline recommended sites. If None, solved dynamically.

    Returns
    -------
    Dict[str, Any]
        Dictionary of stability percentages for all variations.
    """
    logger.info("Starting sensitivity analysis...")
    try:
        # 1. Establish baseline solution
        if recommended_sites is None or recommended_sites.empty:
            logger.info("Computing baseline recommendation for sensitivity comparison...")
            grid_base = _recompute_grid_supply_and_demand(grid_gdf, chargers_gdf, config)
            cands_base = _resolve_candidates(grid_base, candidates_gdf, base_dir)
            if feasibility_filter is not None and substations_gdf is not None:
                try:
                    cands_base = feasibility_filter(cands_base, substations_gdf, config)
                except Exception:
                    pass
            recommended_sites = _execute_solver(None, grid_base, cands_base, config)

        orig_proj = _to_metric_crs(recommended_sites)
        orig_coords = _extract_xy_coords(orig_proj)
        n_orig = len(orig_coords)

        if n_orig == 0:
            logger.warning("Original recommended sites empty; aborting sensitivity analysis.")
            return {"mean_stability_pct": 0.0, "variations": {}}

        orig_h3_set = set()
        if "h3_index" in recommended_sites.columns:
            orig_h3_set = set(recommended_sites["h3_index"].astype(str))
        elif recommended_sites.index.name == "h3_index":
            orig_h3_set = set(recommended_sites.index.astype(str))

        def _compute_stability(var_sites: gpd.GeoDataFrame) -> float:
            if var_sites is None or var_sites.empty:
                return 0.0

            var_proj = _to_metric_crs(var_sites, target_crs=orig_proj.crs)
            var_coords = _extract_xy_coords(var_proj)

            # Check H3 index overlap if available
            var_h3_set = set()
            if "h3_index" in var_sites.columns:
                var_h3_set = set(var_sites["h3_index"].astype(str))
            elif var_sites.index.name == "h3_index":
                var_h3_set = set(var_sites.index.astype(str))

            if orig_h3_set and var_h3_set:
                overlap = len(orig_h3_set.intersection(var_h3_set))
                return float((overlap / min(len(orig_h3_set), len(var_h3_set))) * 100.0)

            # Fallback: spatial matching within 300m
            tree = cKDTree(var_coords)
            nearest_dists, _ = tree.query(orig_coords, k=1)
            retained = np.sum(nearest_dists <= 300.0)
            return float((retained / min(n_orig, len(var_coords))) * 100.0)

        results: Dict[str, Any] = {
            "weight_variations": {},
            "radius_variations": {},
            "n_variations": {},
        }
        all_stabilities: List[float] = []

        # ---------------------------------------------------------
        # A. Feature Weight Perturbations (+/- 20%)
        # ---------------------------------------------------------
        base_weights = dict(config.get("feature_weights", {
            "population": 0.35,
            "poi": 0.25,
            "traffic": 0.25,
            "existing_chargers_penalty": 0.15,
        }))

        for weight_key in ["population", "poi", "traffic", "existing_chargers_penalty"]:
            if weight_key not in base_weights:
                continue

            for delta_pct, label in [(1.20, "+20%"), (0.80, "-20%")]:
                cfg_var = dict(config)
                var_weights = dict(base_weights)
                var_weights[weight_key] = base_weights[weight_key] * delta_pct

                # Normalize weights to sum to 1.0
                total_w = sum(var_weights.values())
                var_weights = {k: v / total_w for k, v in var_weights.items()}
                cfg_var["feature_weights"] = var_weights

                grid_var = _recompute_grid_supply_and_demand(grid_gdf, chargers_gdf, cfg_var)
                cands_var = _resolve_candidates(grid_var, candidates_gdf, base_dir)
                var_sites = _execute_solver(None, grid_var, cands_var, cfg_var)

                stab = _compute_stability(var_sites)
                var_name = f"{weight_key}_{label}"
                results["weight_variations"][var_name] = round(stab, 1)
                all_stabilities.append(stab)

        # ---------------------------------------------------------
        # B. Radius Perturbations (+/- 500m)
        # ---------------------------------------------------------
        base_radius = float(config.get("coverage_radius_m", 1500.0))
        for delta_r, label in [(-500.0, "-500m"), (500.0, "+500m")]:
            new_r = max(500.0, base_radius + delta_r)
            cfg_var = dict(config)
            cfg_var["coverage_radius_m"] = new_r

            grid_var = _recompute_grid_supply_and_demand(grid_gdf, chargers_gdf, cfg_var)
            cands_var = _resolve_candidates(grid_var, candidates_gdf, base_dir)
            var_sites = _execute_solver(None, grid_var, cands_var, cfg_var)

            stab = _compute_stability(var_sites)
            var_name = f"radius_{label}"
            results["radius_variations"][var_name] = round(stab, 1)
            all_stabilities.append(stab)

        # ---------------------------------------------------------
        # C. Station Count Perturbations (+/- 5)
        # ---------------------------------------------------------
        base_n = int(config.get("n_new_stations", 20))
        for delta_n, label in [(-5, "-5"), (5, "+5")]:
            new_n = max(1, base_n + delta_n)
            cfg_var = dict(config)
            cfg_var["n_new_stations"] = new_n

            grid_var = _recompute_grid_supply_and_demand(grid_gdf, chargers_gdf, cfg_var)
            cands_var = _resolve_candidates(grid_var, candidates_gdf, base_dir)
            var_sites = _execute_solver(None, grid_var, cands_var, cfg_var)

            stab = _compute_stability(var_sites)
            var_name = f"stations_{label}"
            results["n_variations"][var_name] = round(stab, 1)
            all_stabilities.append(stab)

        mean_stability = float(np.mean(all_stabilities)) if all_stabilities else 0.0
        results["mean_stability_pct"] = round(mean_stability, 1)

        # Save summary to disk
        base_path = Path(base_dir)
        outputs_dir = base_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        json_path = outputs_dir / "sensitivity_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(
            "Sensitivity analysis complete. Mean stability: %.1f%% across %d variations. Saved to %s",
            mean_stability,
            len(all_stabilities),
            json_path,
        )
        return results

    except Exception as e:
        logger.error("Sensitivity analysis failed: %s", str(e), exc_info=True)
        return {"mean_stability_pct": 0.0, "error": str(e)}
