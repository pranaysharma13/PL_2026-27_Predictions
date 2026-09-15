"""How much do team ratings move during a season, and what that costs you.

The bootstrap in `simulate.py` answers "how uncertain am I about how good
Arsenal are *today*". It does not answer "how good will Arsenal be in March".
Those are different questions. Squads pick up injuries, form turns, managers
get sacked, January happens. A rating fitted in September is a fair estimate of
today and a worse estimate of every match further out.

Ignoring that makes September projections overconfident, which is exactly when
people most want to read them. The fix is to let ratings wander during the
simulation as a random walk, one step per matchweek. The hard part is the step
size.

**An approach that does not work**, recorded so it is not tried again: fit
ratings early and late in each past season, treat the difference as drift, and
subtract bootstrap variance to remove estimation error. Tested against
synthetic seasons with a known step size, it fails badly in both directions.
With a short measurement half-life it reports substantial drift where the truth
is zero; with a long one it compresses real drift to a third of its size. The
bootstrap does not cleanly separate estimation error from genuine movement
under time-decayed weighting, and no half-life fixes it.

**What is used instead** sidesteps latent drift entirely and calibrates against
the thing that actually matters: are the projections honest? For each past
season, stand at matchweek `early_week`, project the final table, and see where
each club's real final points landed inside the predicted distribution. If the
intervals are right, roughly 80% of clubs should land inside their own 10th-to
90th-percentile band. Too few means overconfidence. So try a range of step
sizes and keep whichever gets closest to 80%.

That is slower, but it validates the correction rather than assuming it, and it
fails loudly instead of quietly returning a plausible wrong number.

One thing to be clear about: the number this returns is a calibration knob, not
a measurement of drift. On synthetic seasons with no drift at all it still
settles on a small positive value, because the bootstrap alone leaves the
projections slightly too narrow and this parameter absorbs that as well. That
is the intended behaviour, since the goal is honest intervals rather than a
decomposition of where the uncertainty came from. Do not read the number as
"how much teams change".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import DixonColes

MATCHES_PER_ROUND = 10  # a 20-team division


def schedule_rounds(
    fixtures: list[tuple[str, str]], teams: list[str]
) -> list[list[tuple[str, str]]]:
    """Group remaining fixtures into rounds where no team appears twice.

    The real fixture list isn't needed. What matters for drift is only how far
    into the future a match sits, and a greedy round-robin packing recovers
    that ordering closely enough: each round is one matchweek's worth.
    """
    remaining = list(fixtures)
    rounds: list[list[tuple[str, str]]] = []
    while remaining:
        used: set[str] = set()
        this_round: list[tuple[str, str]] = []
        leftover: list[tuple[str, str]] = []
        for h, a in remaining:
            if h in used or a in used:
                leftover.append((h, a))
            else:
                this_round.append((h, a))
                used.add(h)
                used.add(a)
        rounds.append(this_round)
        remaining = leftover
    return rounds


def _cut_date(matches: pd.DataFrame, season: str, div: str, week: int) -> pd.Timestamp | None:
    """Date just after `week` matchweeks of a season have been played."""
    sub = matches[(matches["season"] == season) & (matches["div"] == div)].sort_values("date")
    n = week * MATCHES_PER_ROUND
    if len(sub) <= n:
        return None
    return sub.iloc[n]["date"]


def _season_setup(
    matches: pd.DataFrame, season: str, div: str, early_week: int, n_boot: int, **model_kwargs
):
    """Everything needed to replay one past season from `early_week` onward."""
    from .data import current_table, remaining_fixtures
    from .simulate import bootstrap_fits

    cut = _cut_date(matches, season, div, early_week)
    if cut is None:
        return None
    season_matches = matches[(matches["season"] == season) & (matches["div"] == div)]
    if len(season_matches) < 380:
        return None

    train = matches[matches["date"] < cut]
    partial = train[(train["season"] == season) & (train["div"] == div)]
    table = current_table(partial, div=div, season=season)
    teams = sorted(set(season_matches["home"]) | set(season_matches["away"]))
    if len(table) != len(teams):
        # A club with no matches in the first few weeks would be missing.
        table = table.set_index("team").reindex(teams).fillna(0).reset_index()
        for c in ("played", "won", "drawn", "lost", "gf", "ga", "points"):
            table[c] = table[c].astype(int)

    played = {(r.home, r.away) for r in partial.itertuples(index=False)}
    fixtures = [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]

    final = current_table(season_matches, div=div, season=season).set_index("team")
    actual = np.array([final.loc[t, "points"] for t in table["team"]], dtype=float)

    try:
        boots = bootstrap_fits(train, n_boot=n_boot, verbose=False, **model_kwargs)
    except (RuntimeError, ValueError):
        return None
    boots = [b for b in boots if all(t in b.teams for t in table["team"])]
    if not boots:
        return None
    return {"season": season, "models": boots, "table": table,
            "fixtures": fixtures, "actual": actual}


def calibrate_drift(
    matches: pd.DataFrame,
    seasons: list[str],
    early_week: int = 4,
    div: str = "E0",
    mode: str = "spread",
    values: tuple[float, ...] | None = None,
    half_life_weeks: float = 15.0,
    n_boot: int = 20,
    n_sims: int = 4000,
    target_coverage: float = 0.80,
    verbose: bool = True,
    **model_kwargs,
) -> dict:
    """Find the setting that makes historical projections honest.

    `mode` picks what is being calibrated:
      "spread"  long-run spread of mean-reverting drift (recommended)
      "sigma"   per-matchweek step of a pure random walk
      "widen"   post-hoc stretch factor on the reported points spread, which
                changes nothing about the dynamics and only widens the answer
    """
    from .simulate import simulate_season

    if mode not in {"spread", "sigma", "widen"}:
        raise ValueError(f"mode must be spread, sigma or widen, got {mode!r}")
    if values is None:
        values = {"spread": (0.0, 0.1, 0.2, 0.3, 0.5),
                  "sigma": (0.0, 0.02, 0.05, 0.08, 0.12),
                  "widen": (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)}[mode]

    setups = []
    for season in seasons:
        if verbose:
            print(f"  preparing {season}")
        setup = _season_setup(matches, season, div, early_week, n_boot, **model_kwargs)
        if setup is None:
            if verbose:
                print(f"    skipped ({season} is incomplete or unusable)")
            continue
        setups.append(setup)
    if not setups:
        raise RuntimeError("no usable seasons for drift calibration")

    rows = []
    for value in values:
        inside, pits = [], []
        for setup in setups:
            res = simulate_season(
                setup["models"], setup["table"], setup["fixtures"],
                n_sims=n_sims, div=div,
                drift_sigma=value if mode == "sigma" else 0.0,
                drift_stationary_sd=value if mode == "spread" else None,
                drift_half_life_weeks=half_life_weeks,
                widen=value if mode == "widen" else 1.0,
                return_draws=True, seed=0)
            draws = res["points_draws"]
            order = {t: i for i, t in enumerate(res["teams"])}
            for k, team in enumerate(setup["table"]["team"]):
                col = draws[:, order[team]]
                actual = setup["actual"][k]
                lo, hi = np.percentile(col, [10, 90])
                inside.append(bool(lo <= actual <= hi))
                pits.append(float((col < actual).mean() + 0.5 * (col == actual).mean()))
        coverage = float(np.mean(inside))
        rows.append({"value": value, "coverage": coverage,
                     "gap": abs(coverage - target_coverage),
                     "mean_pit": float(np.mean(pits)), "n": len(inside)})
        if verbose:
            print(f"  {mode} {value:.3f}: {coverage:.1%} of clubs inside their 10-90 band")

    grid = pd.DataFrame(rows).sort_values("gap").reset_index(drop=True)
    best = grid.iloc[0]
    return {"value": float(best["value"]), "mode": mode,
            "half_life_weeks": half_life_weeks,
            "coverage": float(best["coverage"]),
            "target_coverage": target_coverage, "grid": grid,
            "n_observations": int(best["n"]), "seasons": [s["season"] for s in setups],
            "early_week": early_week}


def rating_spread(model, teams: list[str]) -> float:
    """Spread of attack and defence ratings across a set of teams.

    Used as the long-run bound on drift, so a club's plausible future range
    resembles the range clubs actually occupy rather than growing forever.
    """
    vals = []
    for t in teams:
        if t in model.teams:
            i = model.teams.index(t)
            vals.extend([model.attack[i], model.defence[i]])
    return float(np.std(vals)) if vals else 0.0


def format_drift(res: dict) -> str:
    mode = res["mode"]
    label = {"spread": "long-run spread", "sigma": "drift/week",
             "widen": "widening factor"}[mode]
    subtitle = {
        "spread": f"mean-reverting drift, half-life {res['half_life_weeks']:.0f} matchweeks",
        "sigma": "pure random-walk drift",
        "widen": "post-hoc widening; the simulation dynamics are unchanged",
    }[mode]
    lines = [
        f"calibrated on {', '.join(res['seasons'])}, standing at matchweek "
        f"{res['early_week']} ({res['n_observations']} club-seasons)",
        subtitle,
        "",
        f"{label:>16}{'coverage':>11}{'mean PIT':>11}",
    ]
    for r in res["grid"].sort_values("value").itertuples(index=False):
        mark = "  <-" if r.value == res["value"] else ""
        lines.append(f"{r.value:>16.3f}{r.coverage:>11.1%}{r.mean_pit:>11.3f}{mark}")
    flag = {"spread": "--drift-spread", "sigma": "--drift", "widen": "--widen"}[mode]
    lines += [
        "",
        f"target coverage {res['target_coverage']:.0%}, best {res['coverage']:.1%} "
        f"at {res['value']:.3f}   (pass it with {flag})",
        "",
        "Coverage is the share of clubs whose real final points landed inside their",
        "own 10th-to-90th percentile band. Below target means overconfident.",
        "Mean PIT near 0.5 means the projections are unbiased, not just wide enough.",
    ]
    if res["value"] == res["grid"]["value"].max():
        lines.append("\nWARNING: the best value is the largest tested. Widen the range.")
    return "\n".join(lines)