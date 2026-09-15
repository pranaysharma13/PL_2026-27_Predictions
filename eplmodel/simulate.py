"""Monte Carlo simulation of the rest of the season.

Sample a scoreline for every remaining fixture from the fitted model, add it to
the points already on the board, rank the table, repeat a few tens of thousands
of times, and count how often each outcome happens.

There are two sources of uncertainty here and they are not the same size.

**Match randomness** is that a better team loses sometimes. Sampling scorelines
captures it.

**Rating uncertainty** is that you do not actually know how good each team is.
You estimated it from a finite number of matches. Treating the fitted ratings
as exact makes the projections look more confident than they deserve, and the
effect is worst in September, when the ratings rest on four matchweeks of the
current season. `bootstrap_fits` addresses it: refit the model many times on
resampled data and simulate across the whole set, so the spread of plausible
ratings flows through into the spread of plausible league tables.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .drift import schedule_rounds
from .model import DixonColes


def bootstrap_fits(
    matches: pd.DataFrame,
    n_boot: int = 40,
    seed: int = 0,
    verbose: bool = True,
    **model_kwargs,
) -> list[DixonColes]:
    """Refit the model on `n_boot` resamples of the match data.

    This is a parametric bootstrap over matches: draw len(matches) matches with
    replacement and refit. The spread across the resulting models is an
    estimate of how uncertain the ratings are. Each fit takes a fraction of a
    second, so 40 of them is a matter of seconds.
    """
    rng = np.random.default_rng(seed)
    n = len(matches)
    out: list[DixonColes] = []
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = matches.iloc[idx].sort_values("date").reset_index(drop=True)
        try:
            out.append(DixonColes(**model_kwargs).fit(sample, verbose=False))
        except (ValueError, KeyError):
            continue
        if verbose and (i + 1) % 10 == 0:
            print(f"  bootstrap fit {i + 1}/{n_boot}")
    if not out:
        raise RuntimeError("every bootstrap fit failed")
    return out


def simulate_season(
    model: DixonColes | list[DixonColes],
    table: pd.DataFrame,
    fixtures: list[tuple[str, str]],
    n_sims: int = 20000,
    div: str = "E0",
    rating_jitter: float = 0.0,
    drift_sigma: float = 0.0,
    drift_stationary_sd: float | None = None,
    drift_half_life_weeks: float = 15.0,
    widen: float = 1.0,
    n_blocks: int | None = None,
    return_draws: bool = False,
    seed: int = 0,
) -> dict:
    """Returns a dict with a summary table and the full position distribution.

    `model` may be a single fitted model or a list of bootstrap fits, in which
    case simulations are split across them so the intervals account for
    uncertainty in today's ratings as well as match randomness.

    `drift_sigma` additionally lets ratings wander as the season progresses,
    one step per matchweek. That covers the third source of uncertainty: teams
    genuinely change. Calibrate it with drift.calibrate_drift rather than
    guessing.

    `drift_stationary_sd` switches to mean-reverting drift and is the preferred
    mode. A pure random walk has unbounded variance, so given 34 matchweeks a
    bottom club can drift all the way to title-winning, which real teams do not
    do. Mean reversion lets ratings wander while pulling them back, so the
    spread grows and then settles at this value.

    In this mode the step size is derived rather than given: ratings revert with
    a half-life of `drift_half_life_weeks`, and the step follows from that and
    the target spread. That leaves one free parameter instead of two, and it is
    the one end-of-season coverage can actually identify. `drift_sigma` is
    ignored when this is set.

    `widen` stretches each club's simulated points around its own mean before
    the table is ranked. It is a stated correction, not a mechanism: the
    projections are known to come out about 15% too narrow for reasons that
    drift, shrinkage and the bootstrap between them do not explain, and this
    widens the reported spread to match reality without pretending to know why.
    Because each club is stretched around its own mean, the ordering and the
    correlation between clubs survive, so title and relegation probabilities
    soften rather than becoming incoherent. 1.0 leaves everything untouched.

    `table` must have columns team, points, gf, ga (as produced by
    data.current_table). `fixtures` is a list of (home, away) tuples.
    """
    rng = np.random.default_rng(seed)
    teams = list(table["team"])
    n_t = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    models = list(model) if isinstance(model, (list, tuple)) else [model]
    usable = [m for m in models if all(t in m.teams for t in teams)]
    if not usable:
        raise ValueError("no fitted model covers every team in the table")
    # Fixtures are grouped into matchweeks so drift can accumulate with time.
    # Without drift the ordering is irrelevant, so keep it as one block.
    drifting = drift_sigma > 0 or bool(drift_stationary_sd)
    rounds = schedule_rounds(fixtures, teams) if drifting else [list(fixtures)]

    # Greedy packing leaves stragglers, so it yields a few more rounds than a
    # real fixture list would. Scale the step so the total drift across the
    # season matches the true number of matchweeks rather than the round count.
    ideal_weeks = math.ceil(len(fixtures) / max(1, n_t // 2))
    drift_step = drift_sigma * math.sqrt(ideal_weeks / max(1, len(rounds)))

    # Mean reversion. For an AR(1) step x <- (1-k)x + step*e the long-run
    # variance is step^2 / (2k - k^2). Fixing the reversion half-life pins k,
    # and the step then follows from the target spread. k = 0 is a pure
    # random walk, whose variance grows without bound.
    reversion = 0.0
    if drift_stationary_sd and drift_stationary_sd > 0:
        reversion = 1.0 - 0.5 ** (1.0 / max(drift_half_life_weeks, 0.5))
        per_week = drift_stationary_sd * math.sqrt(2 * reversion - reversion**2)
        drift_step = per_week * math.sqrt(ideal_weeks / max(1, len(rounds)))

    if n_blocks is None:
        n_blocks = max(len(usable), 50) if drifting else len(usable)
    n_blocks = max(1, min(n_blocks, n_sims))
    block_edges = np.linspace(0, n_sims, n_blocks + 1).astype(int)

    points = np.tile(table["points"].to_numpy(dtype=np.int32), (n_sims, 1))
    gf = np.tile(table["gf"].to_numpy(dtype=np.int32), (n_sims, 1))
    ga = np.tile(table["ga"].to_numpy(dtype=np.int32), (n_sims, 1))

    for b in range(n_blocks):
        lo, hi = block_edges[b], block_edges[b + 1]
        if hi <= lo:
            continue
        mdl = usable[b % len(usable)]
        width = mdl.max_goals + 1
        base_a, base_d = mdl.attack.copy(), mdl.defence.copy()
        offset_a = np.zeros_like(base_a)
        offset_d = np.zeros_like(base_d)
        if rating_jitter > 0:
            offset_a += rng.normal(0, rating_jitter, base_a.shape)
            offset_d += rng.normal(0, rating_jitter, base_d.shape)
        n = hi - lo

        for w, this_round in enumerate(rounds):
            if drifting and w > 0:
                # One random-walk step per matchweek, so uncertainty about a
                # match grows with how far away it is.
                keep = 1.0 - reversion
                offset_a = keep * offset_a + rng.normal(0, drift_step, base_a.shape)
                offset_d = keep * offset_d + rng.normal(0, drift_step, base_d.shape)
            mdl.attack = base_a + offset_a
            mdl.defence = base_d + offset_d

            for home, away in this_round:
                cdf = np.cumsum(mdl.score_matrix(home, away, div).ravel())
                cdf[-1] = 1.0
                draw = np.searchsorted(cdf, rng.random(n))
                hg = (draw // width).astype(np.int32)
                ag = (draw % width).astype(np.int32)
                h, a = idx[home], idx[away]
                gf[lo:hi, h] += hg
                ga[lo:hi, h] += ag
                gf[lo:hi, a] += ag
                ga[lo:hi, a] += hg
                home_win = hg > ag
                away_win = ag > hg
                drawn = ~(home_win | away_win)
                points[lo:hi, h] += 3 * home_win + drawn
                points[lo:hi, a] += 3 * away_win + drawn

        mdl.attack, mdl.defence = base_a, base_d

    if widen != 1.0:
        # Stretch around each club's own mean, then round back to whole points
        # so that ties still happen and goal difference still decides them.
        mean_pts = points.mean(axis=0, keepdims=True)
        points = np.rint(mean_pts + widen * (points - mean_pts)).astype(np.int32)
        points = np.clip(points, 0, None)

    gd = gf - ga
    # Rank on points, then goal difference, then goals for, then a coin flip.
    noise = rng.random((n_sims, n_t))
    key = (
        points.astype(np.float64) * 1e9
        + gd.astype(np.float64) * 1e5
        + gf.astype(np.float64) * 1e1
        + noise
    )
    order = np.argsort(-key, axis=1)
    position = np.empty_like(order)
    np.put_along_axis(position, order, np.arange(1, n_t + 1)[None, :].repeat(n_sims, 0), axis=1)

    pos_dist = np.zeros((n_t, n_t))
    for p in range(1, n_t + 1):
        pos_dist[:, p - 1] = (position == p).mean(axis=0)

    summary = pd.DataFrame(
        {
            "team": teams,
            "current_points": table["points"].to_numpy(),
            "proj_points": points.mean(axis=0),
            "pts_10th_pct": np.percentile(points, 10, axis=0),
            "pts_90th_pct": np.percentile(points, 90, axis=0),
            "p_title": pos_dist[:, 0],
            "p_top4": pos_dist[:, :4].sum(axis=1),
            "p_top5": pos_dist[:, :5].sum(axis=1),
            "p_relegated": pos_dist[:, -3:].sum(axis=1),
            "mean_position": position.mean(axis=0),
        }
    ).sort_values("proj_points", ascending=False).reset_index(drop=True)
    summary.index += 1

    pos_df = pd.DataFrame(pos_dist, index=teams, columns=range(1, n_t + 1))
    out = {
        "summary": summary,
        "position_distribution": pos_df,
        "n_sims": n_sims,
        "n_models": len(usable),
        "n_blocks": n_blocks,
        "n_rounds": len(rounds),
        "drift_sigma": drift_sigma,
        "drift_stationary_sd": drift_stationary_sd,
        "drift_reversion": reversion,
        "drift_step": drift_step,
        "widen": widen,
    }
    if return_draws:
        out["points_draws"] = points          # (n_sims, n_teams), column order = teams
        out["teams"] = teams
    return out


def format_summary(result: dict, top: int | None = None) -> str:
    df = result["summary"].copy()
    if top:
        df = df.head(top)
    out = df.assign(
        proj_points=lambda d: d["proj_points"].round(1),
        p_title=lambda d: (d["p_title"] * 100).round(1),
        p_top4=lambda d: (d["p_top4"] * 100).round(1),
        p_top5=lambda d: (d["p_top5"] * 100).round(1),
        p_relegated=lambda d: (d["p_relegated"] * 100).round(1),
    )[["team", "current_points", "proj_points", "pts_10th_pct", "pts_90th_pct",
       "p_title", "p_top4", "p_top5", "p_relegated"]]
    out.columns = ["team", "pts", "proj", "p10", "p90", "title%", "top4%", "top5%", "releg%"]
    return out.to_string()