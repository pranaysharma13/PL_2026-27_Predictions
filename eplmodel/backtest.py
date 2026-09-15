"""Walk-forward evaluation, and the only honest way to know if this works.

The rule: to predict a match on date t, fit only on matches strictly before t.
Never train on a season and score it. The `as_of` argument to DixonColes.fit
enforces this, and this module drives it forward one week at a time.

Metrics:
  RPS        Ranked Probability Score. The primary metric for football, because
             home/draw/away is ordered - calling an away win when it was a draw
             should cost less than calling it when it was a home win. Lower is
             better.
  log loss   Punishes confident errors hard. Lower is better.
  Brier      Multiclass Brier score. Lower is better.

Everything is reported alongside the same metrics computed from bookmaker
closing odds. That comparison is the whole point. If the model is materially
worse than the closing line, it isn't finished. If it matches the line, it is
genuinely good, because the line already contains everyone else's model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .model import DixonColes


# ------------------------------------------------------------------ metrics

def rps(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Ranked Probability Score for ordered outcomes. probs columns: H, D, A."""
    obs = np.zeros_like(probs)
    obs[np.arange(len(outcomes)), outcomes] = 1.0
    cp = np.cumsum(probs, axis=1)[:, :-1]
    co = np.cumsum(obs, axis=1)[:, :-1]
    return ((cp - co) ** 2).sum(axis=1) / (probs.shape[1] - 1)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-12, 1.0)
    return -np.log(p)


def brier(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    obs = np.zeros_like(probs)
    obs[np.arange(len(outcomes)), outcomes] = 1.0
    return ((probs - obs) ** 2).sum(axis=1)


# ------------------------------------------------------- odds to probability

def remove_margin(odds: np.ndarray, method: str = "power") -> np.ndarray:
    """Convert decimal odds to probabilities with the bookmaker margin stripped.

    'proportional' just normalises 1/odds. It is simple and slightly biased,
    because bookmakers load more margin onto longshots.
    'power' finds k with sum((1/o_i)^k) = 1, which distributes the margin more
    realistically and gives a tougher, fairer benchmark.
    """
    inv = 1.0 / odds
    if method == "proportional":
        return inv / inv.sum(axis=1, keepdims=True)
    out = np.empty_like(inv)
    for i, row in enumerate(inv):
        if not np.all(np.isfinite(row)) or row.sum() <= 1.0:
            out[i] = row / row.sum()
            continue
        try:
            k = brentq(lambda k: np.sum(row**k) - 1.0, 0.2, 3.0, xtol=1e-10)
            out[i] = row**k
        except ValueError:
            out[i] = row / row.sum()
    return out / out.sum(axis=1, keepdims=True)


def outcome_index(hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


# ------------------------------------------------------------- walk forward

def walk_forward(
    matches: pd.DataFrame,
    test_season: str,
    test_div: str = "E0",
    refit_every_days: int = 7,
    min_train_matches: int = 400,
    verbose: bool = True,
    **model_kwargs,
) -> pd.DataFrame:
    """Predict every match of `test_season` using only prior data.

    Returns one row per match with model probabilities, market probabilities
    where available, and the realised outcome.
    """
    test = matches[(matches["season"] == test_season) & (matches["div"] == test_div)].copy()
    if test.empty:
        raise ValueError(f"no matches found for season {test_season} in {test_div}")

    dates = sorted(test["date"].unique())
    blocks, current = [], [dates[0]]
    for d in dates[1:]:
        if (pd.Timestamp(d) - pd.Timestamp(current[0])).days >= refit_every_days:
            blocks.append(current)
            current = [d]
        else:
            current.append(d)
    blocks.append(current)

    rows = []
    for bi, block in enumerate(blocks):
        cutoff = pd.Timestamp(block[0])
        train = matches[matches["date"] < cutoff]
        if len(train) < min_train_matches:
            continue
        model = DixonColes(**model_kwargs)
        try:
            model.fit(train, as_of=cutoff)
        except ValueError:
            continue
        chunk = test[test["date"].isin(block)]
        for r in chunk.itertuples(index=False):
            try:
                pred = model.predict(r.home, r.away, test_div)
            except KeyError:
                continue  # a team with no history at all before this date
            rows.append(
                {
                    "date": r.date,
                    "home": r.home,
                    "away": r.away,
                    "hg": r.hg,
                    "ag": r.ag,
                    "m_home": pred["p_home"],
                    "m_draw": pred["p_draw"],
                    "m_away": pred["p_away"],
                    "odds_h": r.odds_h,
                    "odds_d": r.odds_d,
                    "odds_a": r.odds_a,
                }
            )
        if verbose:
            print(f"  block {bi + 1}/{len(blocks)} from {cutoff.date()}: "
                  f"trained on {len(train)} matches, predicted {len(chunk)}")

    return pd.DataFrame(rows)


def evaluate(preds: pd.DataFrame, margin_method: str = "power") -> dict:
    """Aggregate metrics for the model and, where odds exist, the market."""
    out = outcome_index(preds["hg"].to_numpy(), preds["ag"].to_numpy())
    mp = preds[["m_home", "m_draw", "m_away"]].to_numpy()

    result = {
        "n_matches": len(preds),
        "model": {
            "rps": float(rps(mp, out).mean()),
            "log_loss": float(log_loss(mp, out).mean()),
            "brier": float(brier(mp, out).mean()),
            "accuracy": float((mp.argmax(axis=1) == out).mean()),
        },
    }

    has_odds = preds[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)
    if has_odds.any():
        sub = preds[has_odds]
        o = sub[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
        bp = remove_margin(o, margin_method)
        so = outcome_index(sub["hg"].to_numpy(), sub["ag"].to_numpy())
        sm = sub[["m_home", "m_draw", "m_away"]].to_numpy()
        result["market"] = {
            "rps": float(rps(bp, so).mean()),
            "log_loss": float(log_loss(bp, so).mean()),
            "brier": float(brier(bp, so).mean()),
            "accuracy": float((bp.argmax(axis=1) == so).mean()),
            "n_matches": int(has_odds.sum()),
        }
        result["model_on_odds_subset"] = {
            "rps": float(rps(sm, so).mean()),
            "log_loss": float(log_loss(sm, so).mean()),
        }

    # Baseline: the unconditional base rate. If you can't beat this, stop.
    base = np.bincount(out, minlength=3) / len(out)
    bmat = np.tile(base, (len(out), 1))
    result["base_rate"] = {"rps": float(rps(bmat, out).mean()),
                           "log_loss": float(log_loss(bmat, out).mean())}
    return result


def calibration(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Are the stated probabilities honest? Predicted vs realised, by bucket."""
    out = outcome_index(preds["hg"].to_numpy(), preds["ag"].to_numpy())
    p = preds[["m_home", "m_draw", "m_away"]].to_numpy().ravel()
    hit = np.zeros((len(out), 3))
    hit[np.arange(len(out)), out] = 1.0
    hit = hit.ravel()
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        m = b == k
        if m.sum() == 0:
            continue
        rows.append({"bucket": f"{edges[k]:.1f}-{edges[k + 1]:.1f}",
                     "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()),
                     "observed_rate": float(hit[m].mean())})
    return pd.DataFrame(rows)


def format_evaluation(res: dict) -> str:
    lines = [f"matches scored: {res['n_matches']}", ""]
    lines.append(f"{'':22}{'RPS':>9}{'logloss':>10}{'Brier':>9}{'acc':>8}")
    m = res["model"]
    lines.append(f"{'model':22}{m['rps']:>9.4f}{m['log_loss']:>10.4f}{m['brier']:>9.4f}{m['accuracy']:>8.3f}")
    if "market" in res:
        k = res["market"]
        lines.append(f"{'bookmaker close':22}{k['rps']:>9.4f}{k['log_loss']:>10.4f}{k['brier']:>9.4f}{k['accuracy']:>8.3f}")
        d = res["model_on_odds_subset"]["rps"] - k["rps"]
        lines.append("")
        lines.append(f"RPS gap vs market: {d:+.4f}  ({'model better' if d < 0 else 'market better'})")
    b = res["base_rate"]
    lines.append(f"{'base rate':22}{b['rps']:>9.4f}{b['log_loss']:>10.4f}")
    return "\n".join(lines)