#!/usr/bin/env python3
"""Premier League 2026-27 prediction model.

    python run.py fetch-xg                       # scrape xG once, then cache it
    python run.py table
    python run.py ratings
    python run.py predict --home Arsenal --away "Manchester City"
    python run.py simulate --sims 20000
    python run.py backtest --test-season 2025-26 --compare-targets
    python run.py tune --test-season 2025-26

Results and odds come from football-data.co.uk; xG from FBref (or Understat).
Global options go before the subcommand.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

from eplmodel import (
    DixonColes,
    apply_blend,
    bootstrap_fits,
    calibration,
    calibrate_drift,
    format_drift,
    rating_spread,
    evaluate_blend,
    fit_weight,
    format_blend,
    load_upcoming_fixtures,
    current_table,
    evaluate,
    fetch_xg,
    format_evaluation,
    format_summary,
    load_custom,
    load_matches,
    merge_xg,
    remaining_fixtures,
    sanity_check,
    season_label,
    simulate_season,
)

CURRENT_SEASON_START = 2026
CURRENT_SEASON = season_label(CURRENT_SEASON_START)


# ------------------------------------------------------------------- loading

def _xg_cache_path(args) -> str:
    return os.path.join(args.xg_dir, f"xg_raw_{args.xg_source}.csv")


def _load_or_fetch_xg(args, refresh: bool = False) -> pd.DataFrame:
    path = _xg_cache_path(args)
    if os.path.exists(path) and not refresh:
        xg = pd.read_csv(path, parse_dates=["date"])
        print(f"xG loaded from cache: {path} ({len(xg)} matches)")
        return xg
    divisions = ("E0", "E1") if args.include_championship else ("E0",)
    xg = fetch_xg(args.from_season, CURRENT_SEASON_START, divisions=divisions,
                  source=args.xg_source, xg_dir=args.xg_dir)
    os.makedirs(args.xg_dir, exist_ok=True)
    xg.to_csv(path, index=False)
    print(f"xG cached to {path}")
    return xg


def get_data(args, need_xg: bool | None = None) -> pd.DataFrame:
    if args.csv:
        matches = load_custom(args.csv)
    else:
        divisions = ("E0", "E1") if args.include_championship else ("E0",)
        matches, _ = load_matches(args.from_season, CURRENT_SEASON_START,
                                  divisions=divisions, cache_dir=args.cache_dir,
                                  odds=args.odds)

    want_xg = (args.target != "goals") if need_xg is None else need_xg
    if want_xg and args.csv and matches[["xg_h", "xg_a"]].notna().any().any():
        n = int(matches["xg_h"].notna().sum())
        print(f"using xG already present in {args.csv} ({n}/{len(matches)} matches)")
    elif want_xg:
        try:
            xg = _load_or_fetch_xg(args)
            matches = merge_xg(matches, xg)
        except Exception as exc:
            print(f"\nxG unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
            print("falling back to target=goals\n", file=sys.stderr)
            args.target = "goals"
    return matches


def build_model(args, matches: pd.DataFrame, **overrides) -> DixonColes:
    kw = dict(half_life_days=args.half_life, prior_sd=args.prior_sd,
              target=args.target, blend_weight=args.blend_weight)
    kw.update(overrides)
    return DixonColes(**kw).fit(matches, verbose=True)


# ------------------------------------------------------------------ commands

def cmd_fetch_xg(args):
    xg = _load_or_fetch_xg(args, refresh=args.refresh)
    matches = get_data(args, need_xg=False)
    merged = merge_xg(matches, xg)
    print("\nbiggest gaps between goals and xG (a finishing check, and a join check):")
    chk = sanity_check(merged)
    if not chk.empty:
        print(pd.concat([chk.head(5), chk.tail(5)]).to_string())


def cmd_table(args):
    matches = get_data(args, need_xg=False)
    print()
    print(current_table(matches, div="E0", season=args.season).to_string())


def cmd_ratings(args):
    matches = get_data(args)
    model = build_model(args, matches)
    teams = set(current_table(matches, div="E0", season=args.season)["team"])
    r = model.ratings()
    r = r[r["team"].isin(teams)].reset_index(drop=True)
    r.index += 1
    print("\nteam strength, current Premier League clubs only")
    print("(log-goals scale; zero is the average of every team in the fit)\n")
    print(r.round(3).to_string())


def cmd_predict(args):
    matches = get_data(args)
    model = build_model(args, matches)
    p = model.predict(args.home, args.away)
    print(f"\n{p['home']} vs {p['away']}")
    print(f"  expected goals   {p['xg_home']:.2f} - {p['xg_away']:.2f}")
    print(f"  home win         {p['p_home'] * 100:5.1f}%   (fair odds {1 / p['p_home']:.2f})")
    print(f"  draw             {p['p_draw'] * 100:5.1f}%   (fair odds {1 / p['p_draw']:.2f})")
    print(f"  away win         {p['p_away'] * 100:5.1f}%   (fair odds {1 / p['p_away']:.2f})")
    print(f"  over 2.5 goals   {p['p_over_2_5'] * 100:5.1f}%")
    print(f"  both teams score {p['p_btts'] * 100:5.1f}%")
    print("  most likely scores: " + ", ".join(f"{a}-{b} ({q * 100:.1f}%)" for a, b, q in p["top_scores"]))


def cmd_simulate(args):
    matches = get_data(args)
    model = build_model(args, matches)
    if args.bootstrap:
        print(f"\nrefitting on {args.bootstrap} bootstrap resamples for rating uncertainty")
        model = bootstrap_fits(matches, n_boot=args.bootstrap, seed=args.seed,
                               half_life_days=args.half_life, prior_sd=args.prior_sd,
                               target=args.target, blend_weight=args.blend_weight)
        print(f"  {len(model)} usable fits")
    drift_sigma, drift_spread = 0.0, None
    if args.drift_auto:
        done = _completed_seasons(matches)
        base = model[0] if isinstance(model, list) else model
        teams_now = list(current_table(matches, div="E0", season=args.season)["team"])
        anchor = rating_spread(base, teams_now)
        grid = tuple(round(anchor * f, 4) for f in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0))
        print(f"\ncalibrating on {', '.join(done)} (a few minutes). "
              f"League rating spread is {anchor:.3f}; searching multiples of it.")
        info = calibrate_drift(matches, done, early_week=args.drift_early_week,
                               mode="spread", values=grid,
                               half_life_weeks=args.drift_half_life,
                               n_boot=15, n_sims=4000, half_life_days=args.half_life,
                               prior_sd=args.prior_sd, target=args.target,
                               blend_weight=args.blend_weight)
        print()
        print(format_drift(info))
        drift_spread = info["value"]
    elif args.drift_spread is not None:
        drift_spread = args.drift_spread
    elif args.drift is not None:
        drift_sigma = float(args.drift)

    if args.widen_auto:
        done = _completed_seasons(matches)
        print(f"\ncalibrating the widening factor on {', '.join(done)} (a few minutes)")
        info = calibrate_drift(matches, done, early_week=args.drift_early_week,
                               mode="widen", n_boot=15, n_sims=4000,
                               half_life_days=args.half_life, prior_sd=args.prior_sd,
                               target=args.target, blend_weight=args.blend_weight)
        print()
        print(format_drift(info))
        args.widen = info["value"]

    if args.widen != 1.0:
        print(f"\nwidening reported spread by {args.widen:.2f}x "
              "(a stated correction, not a mechanism)")
    if drift_spread:
        print(f"\nmean-reverting drift: long-run spread {drift_spread:.3f}, "
              f"half-life {args.drift_half_life:.0f} matchweeks")
    elif drift_sigma:
        print(f"\npure random walk drift: sigma {drift_sigma:.4f} per matchweek")

    table = current_table(matches, div="E0", season=args.season)
    teams = list(table["team"])
    if len(teams) != 20:
        print(f"warning: found {len(teams)} teams in {args.season}, expected 20", file=sys.stderr)
    fixtures = remaining_fixtures(matches, teams, div="E0", season=args.season)
    print(f"\n{len(fixtures)} fixtures remaining, running {args.sims} simulations")
    res = simulate_season(model, table, fixtures, n_sims=args.sims,
                          rating_jitter=args.jitter, drift_sigma=drift_sigma,
                          drift_stationary_sd=drift_spread,
                          drift_half_life_weeks=args.drift_half_life,
                          widen=args.widen, seed=args.seed)
    print()
    print(format_summary(res))
    if args.out:
        res["summary"].to_csv(args.out, index=False)
        print(f"\nwritten to {args.out}")


def cmd_backtest(args):
    from eplmodel import walk_forward

    season = args.test_season or args.season
    matches = get_data(args)
    targets = ["goals", "xg", "blend"] if args.compare_targets else [args.target]

    results = {}
    for tgt in targets:
        print(f"\nwalk-forward backtest on {season}, target={tgt}")
        preds = walk_forward(matches, test_season=season, test_div="E0",
                             refit_every_days=args.refit_every,
                             verbose=not args.compare_targets,
                             half_life_days=args.half_life, prior_sd=args.prior_sd,
                             target=tgt, blend_weight=args.blend_weight)
        res = evaluate(preds)
        results[tgt] = (res, preds)
        print(format_evaluation(res))

    if args.compare_targets:
        print("\n" + "=" * 52)
        print(f"{'target':10}{'RPS':>10}{'logloss':>11}{'Brier':>10}{'acc':>9}")
        for tgt, (res, _) in results.items():
            m = res["model"]
            print(f"{tgt:10}{m['rps']:>10.4f}{m['log_loss']:>11.4f}{m['brier']:>10.4f}{m['accuracy']:>9.3f}")
        mk = next(iter(results.values()))[0].get("market")
        if mk:
            print(f"{'market':10}{mk['rps']:>10.4f}{mk['log_loss']:>11.4f}{mk['brier']:>10.4f}{mk['accuracy']:>9.3f}")

    best = args.target if not args.compare_targets else min(results, key=lambda t: results[t][0]["model"]["rps"])
    print(f"\ncalibration ({best})")
    print(calibration(results[best][1]).round(3).to_string(index=False))
    if args.out:
        results[best][1].to_csv(args.out, index=False)
        print(f"\npredictions written to {args.out}")


def _completed_seasons(matches: pd.DataFrame, div: str = "E0") -> list[str]:
    """Seasons with a full 380-match programme, so drift can be measured end to end."""
    counts = matches[matches["div"] == div].groupby("season").size()
    return sorted(counts[counts >= 380].index)


def cmd_drift(args):
    matches = get_data(args)
    seasons = ([s.strip() for s in args.seasons.split(",")] if args.seasons
               else _completed_seasons(matches))
    values = tuple(float(v) for v in args.values.split(",")) if args.values else None
    if values is None and args.mode == "spread":
        base = build_model(args, matches)
        teams = list(current_table(matches, div="E0", season=args.season)["team"])
        anchor = rating_spread(base, teams)
        values = tuple(round(anchor * f, 4) for f in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0))
        print(f"\nleague rating spread is {anchor:.3f}; searching multiples of it")
    print(f"\ncalibrating {args.mode} on {', '.join(seasons)}")
    info = calibrate_drift(matches, seasons, early_week=args.early_week,
                           mode=args.mode, values=values,
                           half_life_weeks=args.half_life_weeks,
                           n_boot=args.n_boot, n_sims=args.sims,
                           half_life_days=args.half_life, prior_sd=args.prior_sd,
                           target=args.target, blend_weight=args.blend_weight)
    print()
    print(format_drift(info))


def cmd_blend(args):
    from eplmodel import walk_forward

    matches = get_data(args)
    kw = dict(half_life_days=args.half_life, prior_sd=args.prior_sd,
              target=args.target, blend_weight=args.blend_weight)

    fit_seasons = [s.strip() for s in args.fit_seasons.split(",")]
    print(f"\nfitting the blend weight on {', '.join(fit_seasons)}")
    frames = [walk_forward(matches, test_season=s, test_div="E0",
                           refit_every_days=args.refit_every, verbose=False, **kw)
              for s in fit_seasons]
    train = pd.concat(frames, ignore_index=True)
    info = fit_weight(train, method=args.pooling)
    lo, hi = info["plausible_range"]
    print(f"  weight {info['weight']:.3f} on the model, from {info['n_matches']} matches")
    print(f"  settings within 0.0001 RPS of the optimum: {lo:.2f} to {hi:.2f}")

    test_season = args.test_season or args.season
    print(f"\nevaluating on {test_season}, which the weight has not seen")
    test = walk_forward(matches, test_season=test_season, test_div="E0",
                        refit_every_days=args.refit_every, verbose=False, **kw)
    print()
    print(format_blend(evaluate_blend(test, info["weight"], method=args.pooling)))
    if args.out:
        apply_blend(test, info["weight"], method=args.pooling).to_csv(args.out, index=False)
        print(f"\npredictions written to {args.out}")


def cmd_card(args):
    from eplmodel import walk_forward

    matches = get_data(args)
    model = build_model(args, matches)
    kw = dict(half_life_days=args.half_life, prior_sd=args.prior_sd,
              target=args.target, blend_weight=args.blend_weight)

    weight = args.weight
    if weight is None:
        seasons = [s.strip() for s in args.fit_seasons.split(",")]
        print(f"fitting the blend weight on {', '.join(seasons)}")
        frames = [walk_forward(matches, test_season=s, test_div="E0",
                               refit_every_days=14, verbose=False, **kw) for s in seasons]
        weight = fit_weight(pd.concat(frames, ignore_index=True), method=args.pooling)["weight"]
        print(f"  weight {weight:.3f} on the model")

    fixtures = load_upcoming_fixtures(divisions=("E0",), odds=args.odds)
    if fixtures.empty:
        print("no upcoming Premier League fixtures in the feed")
        return

    # The feed is a rolling window that still carries the round just played.
    # Drop anything already in the results, so the card only shows real fixtures.
    played = {
        (r.home, r.away)
        for r in matches[(matches["div"] == "E0") & (matches["season"] == args.season)].itertuples(index=False)
    }
    before = len(fixtures)
    fixtures = fixtures[[(h, a) not in played for h, a in zip(fixtures["home"], fixtures["away"])]]
    if before > len(fixtures):
        print(f"  dropped {before - len(fixtures)} fixtures already in the results")
    if fixtures.empty:
        print("every fixture in the feed has already been played; nothing to price yet")
        return

    rows = []
    for r in fixtures.itertuples(index=False):
        try:
            p = model.predict(r.home, r.away, "E0")
        except KeyError as exc:
            print(f"skipping {r.home} v {r.away}: {exc}")
            continue
        rows.append({"date": r.date, "home": r.home, "away": r.away,
                     "hg": 0, "ag": 0,
                     "m_home": p["p_home"], "m_draw": p["p_draw"], "m_away": p["p_away"],
                     "odds_h": r.odds_h, "odds_d": r.odds_d, "odds_a": r.odds_a,
                     "xg_home": p["xg_home"], "xg_away": p["xg_away"]})
    if not rows:
        print("no fixtures could be priced")
        return

    card = apply_blend(pd.DataFrame(rows), weight, method=args.pooling)
    show = pd.DataFrame({
        "date": card["date"].dt.strftime("%a %d %b"),
        "fixture": card["home"] + " v " + card["away"],
        "xG": card["xg_home"].round(2).astype(str) + "-" + card["xg_away"].round(2).astype(str),
        "mH": (card["m_home"] * 100).round(1),
        "mD": (card["m_draw"] * 100).round(1),
        "mA": (card["m_away"] * 100).round(1),
        "H": (card["b_home"] * 100).round(1),
        "D": (card["b_draw"] * 100).round(1),
        "A": (card["b_away"] * 100).round(1),
        "src": np.where(card["blended"], "blend", "model"),
    })
    print()
    print(show.to_string(index=False))
    print("\nmH/mD/mA are the model alone. H/D/A are what you should use.")
    if weight <= 0.001:
        print(f"The fitted weight is {weight:.3f}, so H/D/A are entirely the market's numbers,")
        print("not yours. Compare them against mH/mD/mA to see where you disagree.")
    elif weight >= 0.999:
        print("The fitted weight is 1.000, so H/D/A are the model alone.")
    print("'model' rows had no odds in the feed and are unblended either way.")
    if args.out:
        card.to_csv(args.out, index=False)
        print(f"written to {args.out}")


def cmd_tune(args):
    from eplmodel import walk_forward

    season = args.test_season or args.season
    matches = get_data(args)
    half_lives = [float(v) for v in args.half_lives.split(",")]
    prior_sds = [float(v) for v in args.prior_sds.split(",")]
    print(f"\ngrid search on {season}, target={args.target}: {len(half_lives) * len(prior_sds)} combinations")
    print("each combination refits once per matchweek, so this takes a while\n")
    rows = []
    for hl, sd in itertools.product(half_lives, prior_sds):
        preds = walk_forward(matches, test_season=season, test_div="E0",
                             refit_every_days=args.refit_every, verbose=False,
                             half_life_days=hl, prior_sd=sd,
                             target=args.target, blend_weight=args.blend_weight)
        res = evaluate(preds)
        rows.append({"half_life": hl, "prior_sd": sd,
                     "rps": res["model"]["rps"], "log_loss": res["model"]["log_loss"]})
        print(f"  half_life={hl:>6.0f}  prior_sd={sd:.2f}  RPS={res['model']['rps']:.4f}")
    grid = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    print("\nbest settings first:")
    print(grid.round(4).to_string(index=False))

    best = grid.iloc[0]
    edges = []
    if best["half_life"] in (min(half_lives), max(half_lives)) and len(half_lives) > 1:
        edges.append(f"half_life={best['half_life']:.0f}")
    if best["prior_sd"] in (min(prior_sds), max(prior_sds)) and len(prior_sds) > 1:
        edges.append(f"prior_sd={best['prior_sd']:.2f}")
    if edges:
        print(f"\nWARNING: the best setting sits at the edge of the grid ({', '.join(edges)}).")
        print("The real optimum is probably outside it. Widen the search in that direction,")
        print("e.g. --half-lives 60,90,120,180 --prior-sds 0.5,0.8,1.2,2.0")

    print("\nNote: tuning on one season then reporting that season's score overstates "
          "performance. Tune on one season, report on another. This also optimises the")
    print("average matchweek, which is not the same as optimising matchweek 4.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-season", type=int, default=2019,
                    help="earliest season start year to load (default 2019)")
    ap.add_argument("--season", default=CURRENT_SEASON, help=f"season to predict (default {CURRENT_SEASON})")
    ap.add_argument("--include-championship", action="store_true", default=True,
                    help="fit on the Championship too, so promoted clubs have a real prior")
    ap.add_argument("--no-championship", dest="include_championship", action="store_false")
    ap.add_argument("--csv", help="use your own match CSV instead of downloading")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--odds", choices=["closing", "opening"], default="closing",
                    help="closing is the sharpest benchmark and includes team-sheet "
                         "information; opening is what you could actually act on days ahead")
    ap.add_argument("--half-life", type=float, default=240.0, help="time decay half-life in days")
    ap.add_argument("--prior-sd", type=float, default=0.40, help="shrinkage; smaller pulls harder to average")
    ap.add_argument("--target", choices=["goals", "xg", "blend"], default="blend",
                    help="what the ratings are fitted on (default blend)")
    ap.add_argument("--blend-weight", type=float, default=0.70,
                    help="weight on xG when target=blend (default 0.70)")
    ap.add_argument("--xg-source", choices=["fbref", "understat"], default="understat",
                    help="understat needs no browser but has no Championship xG; "
                         "fbref covers more but is fragile to scrape")
    ap.add_argument("--xg-dir", default="xg_cache")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch-xg", help="scrape xG and cache it")
    p.add_argument("--refresh", action="store_true", help="ignore the cache and re-scrape")
    p.set_defaults(func=cmd_fetch_xg)

    sub.add_parser("table").set_defaults(func=cmd_table)
    sub.add_parser("ratings").set_defaults(func=cmd_ratings)

    p = sub.add_parser("predict")
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("simulate")
    p.add_argument("--sims", type=int, default=20000)
    p.add_argument("--bootstrap", type=int, default=0, metavar="N",
                   help="refit on N resamples so the intervals include rating uncertainty; "
                        "40 is a reasonable start")
    p.add_argument("--drift-spread", type=float, metavar="SD",
                   help="mean-reverting drift: the long-run spread ratings settle at. "
                        "This is the recommended knob; get a value from 'drift'")
    p.add_argument("--drift-auto", action="store_true",
                   help="calibrate the spread against past seasons first")
    p.add_argument("--drift-half-life", type=float, default=15.0,
                   help="matchweeks for drift to reach half its long-run spread")
    p.add_argument("--drift", metavar="SIGMA",
                   help="pure random walk instead: per-matchweek step. Not recommended, "
                        "since a bottom club can drift all the way to title-winning")
    p.add_argument("--drift-early-week", type=int, default=4)
    p.add_argument("--widen", type=float, default=1.0, metavar="K",
                   help="stretch the reported points spread by K. The projections are "
                        "known to run about 15%% narrow for unexplained reasons; this "
                        "corrects the reported numbers without pretending to know why")
    p.add_argument("--widen-auto", action="store_true",
                   help="calibrate the widening factor against past seasons first")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="crude alternative to --bootstrap; per-simulation rating noise")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("backtest")
    p.add_argument("--test-season", help="season to score, e.g. 2025-26")
    p.add_argument("--refit-every", type=int, default=7)
    p.add_argument("--compare-targets", action="store_true",
                   help="run goals, xg and blend side by side")
    p.add_argument("--out")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("drift", help="calibrate rating drift against past seasons")
    p.add_argument("--seasons", help="comma-separated; defaults to every completed season")
    p.add_argument("--early-week", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=20)
    p.add_argument("--sims", type=int, default=4000)
    p.add_argument("--mode", choices=["spread", "sigma", "widen"], default="spread",
                   help="spread: mean-reverting drift. sigma: pure random walk. "
                        "widen: post-hoc stretch of the reported spread")
    p.add_argument("--values", help="comma-separated values to try")
    p.add_argument("--half-life-weeks", type=float, default=15.0)
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("blend", help="fit and evaluate a model/market blend")
    p.add_argument("--fit-seasons", default="2023-24,2024-25",
                   help="seasons used to fit the weight; must exclude the test season")
    p.add_argument("--test-season", help="season to report on, e.g. 2025-26")
    p.add_argument("--pooling", choices=["log", "linear"], default="log")
    p.add_argument("--refit-every", type=int, default=7)
    p.add_argument("--out")
    p.set_defaults(func=cmd_blend)

    p = sub.add_parser("card", help="price the upcoming round, blended where odds exist")
    p.add_argument("--weight", type=float, help="skip fitting and use this model weight")
    p.add_argument("--fit-seasons", default="2023-24,2024-25")
    p.add_argument("--pooling", choices=["log", "linear"], default="log")
    p.add_argument("--out")
    p.set_defaults(func=cmd_card)

    p = sub.add_parser("tune")
    p.add_argument("--test-season", help="season to tune on, e.g. 2025-26")
    p.add_argument("--refit-every", type=int, default=14)
    p.add_argument("--half-lives", default="90,120,180,240,360",
                   help="comma-separated half-lives in days to search")
    p.add_argument("--prior-sds", default="0.35,0.50,0.80,1.20",
                   help="comma-separated prior SDs to search")
    p.set_defaults(func=cmd_tune)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()