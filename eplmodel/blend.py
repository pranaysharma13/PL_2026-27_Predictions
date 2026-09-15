"""Combining model probabilities with the bookmaker's.

The closing line is hard to beat because it aggregates everyone else's model
plus information yours has no access to: injuries, lineups, transfer news,
whether a manager is about to be sacked. Your model, meanwhile, sees seven
seasons of shot-level performance in a consistent way that no individual bettor
does. Neither dominates, so a weighted combination usually beats both.

Two ways to combine:

**Linear pooling** is the plain weighted average, `w*model + (1-w)*market`. It
is conservative: the result can never be more extreme than both inputs.

**Logarithmic pooling** takes a weighted geometric mean, `model^w * market^(1-w)`
renormalised. When both sources agree the result is sharper than either, which
is usually the right behaviour: two independent estimates agreeing is stronger
evidence than one. It also handles disagreement more gracefully, pulling toward
whichever source is less confident rather than splitting the difference. It is
the better default.

The weight is fitted, not chosen. Fit it on seasons the model was never tuned
on, and never on the season you report.

What the fitted weight tells you is worth as much as the accuracy gain. Near
0.5 means your model contributes real independent signal. Near 0.1 means it is
mostly redundant next to the market, which is useful to know plainly rather
than to discover slowly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .backtest import brier, log_loss, outcome_index, remove_margin, rps

PROB_COLS = ["m_home", "m_draw", "m_away"]
ODDS_COLS = ["odds_h", "odds_d", "odds_a"]


def blend_probabilities(
    p_model: np.ndarray, p_market: np.ndarray, weight: float, method: str = "log"
) -> np.ndarray:
    """Combine two probability matrices. `weight` is the weight on the model."""
    if method == "linear":
        out = weight * p_model + (1.0 - weight) * p_market
    elif method == "log":
        m = np.clip(p_model, 1e-12, 1.0)
        k = np.clip(p_market, 1e-12, 1.0)
        out = np.exp(weight * np.log(m) + (1.0 - weight) * np.log(k))
    else:
        raise ValueError(f"method must be 'linear' or 'log', got {method!r}")
    return out / out.sum(axis=1, keepdims=True)


def _split(preds: pd.DataFrame, margin_method: str = "power"):
    """Rows that have odds, as (model probs, market probs, outcomes)."""
    has = preds[ODDS_COLS].notna().all(axis=1)
    sub = preds[has]
    if sub.empty:
        raise ValueError("no rows with bookmaker odds; cannot fit a blend weight")
    p_model = sub[PROB_COLS].to_numpy(dtype=float)
    p_market = remove_margin(sub[ODDS_COLS].to_numpy(dtype=float), margin_method)
    out = outcome_index(sub["hg"].to_numpy(), sub["ag"].to_numpy())
    return p_model, p_market, out, has


def fit_weight(
    preds: pd.DataFrame,
    method: str = "log",
    metric: str = "rps",
    margin_method: str = "power",
) -> dict:
    """Find the model weight that minimises the chosen metric.

    `preds` comes from backtest.walk_forward. Use seasons you will not report on.
    """
    p_model, p_market, out, _ = _split(preds, margin_method)
    score = {"rps": rps, "log_loss": log_loss, "brier": brier}[metric]

    def objective(w):
        return float(score(blend_probabilities(p_model, p_market, w, method), out).mean())

    res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded",
                          options={"xatol": 1e-4})
    w = float(res.x)

    # A flat curve means the weight is barely identified; report the range that
    # sits within one part in ten thousand of the optimum so that is visible.
    grid = np.linspace(0.0, 1.0, 101)
    curve = np.array([objective(g) for g in grid])
    near = grid[curve <= curve.min() + 1e-4]

    return {
        "weight": w,
        "method": method,
        "metric": metric,
        "n_matches": len(out),
        "score_at_optimum": float(res.fun),
        "score_model_only": objective(1.0),
        "score_market_only": objective(0.0),
        "plausible_range": (float(near.min()), float(near.max())),
    }


def apply_blend(
    preds: pd.DataFrame, weight: float, method: str = "log", margin_method: str = "power"
) -> pd.DataFrame:
    """Add blended probabilities. Rows without odds keep the model's own.

    This is what makes one code path serve both uses. Next weekend's fixtures
    have odds and get the blend; a fixture in April has none, the weight
    silently becomes 1, and the raw model comes through unchanged.
    """
    out = preds.copy()
    for c in ("b_home", "b_draw", "b_away"):
        out[c] = np.nan

    has = out[ODDS_COLS].notna().all(axis=1)
    out.loc[~has, ["b_home", "b_draw", "b_away"]] = out.loc[~has, PROB_COLS].to_numpy()

    if has.any():
        p_model = out.loc[has, PROB_COLS].to_numpy(dtype=float)
        p_market = remove_margin(out.loc[has, ODDS_COLS].to_numpy(dtype=float), margin_method)
        out.loc[has, ["b_home", "b_draw", "b_away"]] = blend_probabilities(
            p_model, p_market, weight, method
        )
    out["blended"] = has
    return out


def evaluate_blend(
    preds: pd.DataFrame, weight: float, method: str = "log", margin_method: str = "power"
) -> dict:
    """Score model, market and blend on the same matches, for a fair comparison."""
    p_model, p_market, out, _ = _split(preds, margin_method)
    p_blend = blend_probabilities(p_model, p_market, weight, method)

    def row(p):
        return {
            "rps": float(rps(p, out).mean()),
            "log_loss": float(log_loss(p, out).mean()),
            "brier": float(brier(p, out).mean()),
            "accuracy": float((p.argmax(axis=1) == out).mean()),
        }

    return {
        "n_matches": len(out),
        "weight": weight,
        "method": method,
        "model": row(p_model),
        "market": row(p_market),
        "blend": row(p_blend),
    }


def format_blend(res: dict) -> str:
    lines = [
        f"blend weight {res['weight']:.3f} on the model, {1 - res['weight']:.3f} on the market"
        f"  ({res['method']} pooling, {res['n_matches']} matches)",
        "",
        f"{'':10}{'RPS':>10}{'logloss':>11}{'Brier':>10}{'acc':>9}",
    ]
    for name in ("model", "market", "blend"):
        r = res[name]
        lines.append(f"{name:10}{r['rps']:>10.4f}{r['log_loss']:>11.4f}"
                     f"{r['brier']:>10.4f}{r['accuracy']:>9.3f}")
    gain_m = res["model"]["rps"] - res["blend"]["rps"]
    gain_k = res["market"]["rps"] - res["blend"]["rps"]
    lines += [
        "",
        f"blend beats the model by {gain_m:+.4f} RPS and the market by {gain_k:+.4f}",
    ]
    if gain_k <= 0:
        lines.append("The blend does not beat the market here. Treat the market as your number.")
    return "\n".join(lines)