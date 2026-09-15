"""A Dixon-Coles match model and season simulator for the Premier League."""

from .blend import (
    apply_blend,
    blend_probabilities,
    evaluate_blend,
    fit_weight,
    format_blend,
)
from .drift import calibrate_drift, format_drift, rating_spread, schedule_rounds
from .data import (
    current_table,
    load_custom,
    load_matches,
    load_upcoming_fixtures,
    remaining_fixtures,
    season_label,
)
from .model import DixonColes, outcome_probabilities
from .simulate import bootstrap_fits, format_summary, simulate_season
from .backtest import calibration, evaluate, format_evaluation, walk_forward
from .xg import fetch_xg, merge_xg, sanity_check

__all__ = [
    "load_matches",
    "load_custom",
    "current_table",
    "remaining_fixtures",
    "season_label",
    "DixonColes",
    "outcome_probabilities",
    "simulate_season",
    "format_summary",
    "walk_forward",
    "evaluate",
    "calibration",
    "format_evaluation",
    "fetch_xg",
    "merge_xg",
    "sanity_check",
    "load_upcoming_fixtures",
    "bootstrap_fits",
    "fit_weight",
    "apply_blend",
    "evaluate_blend",
    "blend_probabilities",
    "format_blend",
    "calibrate_drift",
    "format_drift",
    "schedule_rounds",
    "rating_spread",
]