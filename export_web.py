#!/usr/bin/env python3
"""Generate web/data.json for the website.

    python export_web.py
    python export_web.py --sims 30000 --bootstrap 60

Everything the site shows is precomputed here, so the page is a static file
reading JSON. No server, no API, nothing to keep running. Push web/ to GitHub
Pages and it works.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np

from eplmodel import (
    DixonColes, bootstrap_fits, current_table, load_matches, merge_xg,
    rating_spread, remaining_fixtures, sanity_check, simulate_season,
)

CURRENT_START = 2026
SEASON = "2026-27"
SHORT = {
    "Manchester City": "Man City", "Manchester United": "Man Utd",
    "Brighton & Hove Albion": "Brighton", "Tottenham Hotspur": "Spurs",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Forest",
    "AFC Bournemouth": "Bournemouth", "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich", "Coventry City": "Coventry", "Hull City": "Hull",
    "Crystal Palace": "Palace", "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham", "Sheffield United": "Sheffield Utd",
    "Leicester City": "Leicester", "Luton Town": "Luton",
}
short = lambda t: SHORT.get(t, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--bootstrap", type=int, default=40)
    ap.add_argument("--widen", type=float, default=1.5)
    ap.add_argument("--half-life", type=float, default=240.0)
    ap.add_argument("--prior-sd", type=float, default=0.40)
    ap.add_argument("--out", default="web/data.json")
    a = ap.parse_args()

    matches, _ = load_matches(2019, CURRENT_START, divisions=("E0", "E1"))
    xg_cache = "xg_cache/xg_raw_understat.csv"
    if os.path.exists(xg_cache):
        import pandas as pd
        matches = merge_xg(matches, pd.read_csv(xg_cache, parse_dates=["date"]))

    kw = dict(half_life_days=a.half_life, prior_sd=a.prior_sd, target="blend")
    model = DixonColes(**kw).fit(matches, verbose=True)
    table = current_table(matches, div="E0", season=SEASON)
    teams = list(table["team"])
    fixtures = remaining_fixtures(matches, teams, "E0", SEASON)

    print(f"\nrefitting on {a.bootstrap} resamples")
    models = bootstrap_fits(matches, n_boot=a.bootstrap, verbose=False, **kw)
    print(f"simulating {a.sims} seasons")
    res = simulate_season(models, table, fixtures, n_sims=a.sims, widen=a.widen)
    summary = res["summary"].set_index("team")
    posdist = res["position_distribution"]

    ratings = model.ratings()
    ratings = ratings[ratings["team"].isin(teams)]

    fin = sanity_check(matches)
    fin = fin.reindex(fin["goals_minus_xg"].abs().sort_values(ascending=False).index).head(12)

    data = {
        "generated": date.today().isoformat(),
        "season": SEASON,
        "matchweek": int(table["played"].max()),
        "sample": False,
        "meta": {
            "matches": len(matches),
            "xg_matches": int(matches["xg_h"].notna().sum()) if "xg_h" in matches else 0,
            "seasons": f"2019-20 to {SEASON}",
            "sims": a.sims, "bootstrap": len(models), "widen": a.widen,
        },
        "table": [
            {"team": r.team, "short": short(r.team), "played": int(r.played),
             "won": int(r.won), "drawn": int(r.drawn), "lost": int(r.lost),
             "gf": int(r.gf), "ga": int(r.ga), "gd": int(r.gd), "points": int(r.points)}
            for r in table.itertuples(index=False)
        ],
        "projections": [
            {"team": t, "short": short(t),
             "proj": round(float(summary.loc[t, "proj_points"]), 1),
             "p10": int(summary.loc[t, "pts_10th_pct"]),
             "p90": int(summary.loc[t, "pts_90th_pct"]),
             "title": round(float(summary.loc[t, "p_title"]) * 100, 1),
             "top4": round(float(summary.loc[t, "p_top4"]) * 100, 1),
             "top5": round(float(summary.loc[t, "p_top5"]) * 100, 1),
             "releg": round(float(summary.loc[t, "p_relegated"]) * 100, 1),
             "positions": [round(float(v), 5) for v in posdist.loc[t]]}
            for t in summary.sort_values("proj_points", ascending=False).index
        ],
        # Pricing any fixture in the browser needs only these numbers.
        "ratings": {
            "home_adv": round(float(model.home_adv), 4),
            "rho": round(float(model.rho), 4),
            "intercept": round(float(model.div_intercept[model.divisions.index("E0")]), 4),
            "spread": round(rating_spread(model, teams), 4),
            "clubs": [{"team": r.team, "short": short(r.team),
                       "attack": round(float(r.attack), 4),
                       "defence": round(float(r.defence), 4)}
                      for r in ratings.itertuples(index=False)],
        },
        "finishing": [
            {"team": t, "short": short(t), "diff": round(float(row.goals_minus_xg), 1)}
            for t, row in fin.iterrows()
        ],
        # Filled in by hand from backtest and drift runs; they change rarely.
        "backtest": [
            {"season": "2023-24", "goals": 0.1957, "xg": 0.1935, "blend": 0.1926,
             "market": 0.1802, "base": 0.2337},
            {"season": "2024-25", "goals": 0.2081, "xg": 0.2013, "blend": 0.2023,
             "market": 0.1962, "base": 0.2341},
            {"season": "2025-26", "goals": 0.2120, "xg": 0.2090, "blend": 0.2088,
             "market": 0.2048, "base": 0.2273},
        ],
        "calibration": [
            {"widen": 1.0, "coverage": 68.6}, {"widen": 1.1, "coverage": 69.3},
            {"widen": 1.2, "coverage": 72.1}, {"widen": 1.3, "coverage": 73.6},
            {"widen": 1.4, "coverage": 75.0}, {"widen": 1.5, "coverage": 79.3},
            {"widen": 1.6, "coverage": 83.6}, {"widen": 1.7, "coverage": 86.4},
            {"widen": 1.8, "coverage": 90.7},
        ],
    }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"\nwrote {a.out} ({os.path.getsize(a.out) / 1024:.0f} KB)")
    print("Open web/index.html, or push web/ to GitHub Pages.")


if __name__ == "__main__":
    main()