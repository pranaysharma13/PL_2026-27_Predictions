#!/usr/bin/env python3
"""Offline smoke test: builds synthetic data and runs the whole pipeline.

Run this first to confirm the install works without touching the network:

    python smoke_test.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from eplmodel import (
    DixonColes,
    apply_blend,
    calibrate_drift,
    bootstrap_fits,
    evaluate_blend,
    fit_weight,
    current_table,
    evaluate,
    format_evaluation,
    format_summary,
    remaining_fixtures,
    simulate_season,
    walk_forward,
)

TEAMS = [f"Team {chr(65 + i)}" for i in range(20)]


def synthetic(seasons=("2023-24", "2024-25", "2025-26", "2026-27"), played_now=40, seed=7):
    rng = np.random.default_rng(seed)
    att = {t: rng.normal(0, 0.30) for t in TEAMS}
    dfn = {t: rng.normal(0, 0.25) for t in TEAMS}
    rows = []
    for si, season in enumerate(seasons):
        fixtures = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        rng.shuffle(fixtures)
        if season == seasons[-1]:
            fixtures = fixtures[:played_now]
        d0 = pd.Timestamp(f"{2023 + si}-08-15")
        step = 260 / max(len(fixtures), 1)
        for i, (h, a) in enumerate(fixtures):
            lam = math.exp(0.10 + 0.25 + att[h] - dfn[a])
            mu = math.exp(0.10 + att[a] - dfn[h])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            # xG: a tighter measurement of the same underlying rate
            xgh, xga = rng.gamma(6.0, lam / 6.0), rng.gamma(6.0, mu / 6.0)
            p = np.array([0.44, 0.26, 0.30]) + rng.normal(0, 0.03, 3)
            p = np.clip(p, 0.05, None)
            p /= p.sum()
            o = 1 / (p * 1.05)
            rows.append(
                dict(date=d0 + pd.Timedelta(days=int(i * step)), div="E0", season=season,
                     season_start=2023 + si, home=h, away=a, hg=hg, ag=ag,
                     odds_h=o[0], odds_d=o[1], odds_a=o[2], xg_h=xgh, xg_a=xga)
            )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main():
    m = synthetic()
    print(f"synthetic data: {len(m)} matches\n")

    model = DixonColes(half_life_days=300, prior_sd=0.5, target="blend").fit(m, verbose=True)
    print("\ntop 5 by rating:")
    print(model.ratings().head(5).round(3).to_string(index=False))

    print("\nsample match:")
    p = model.predict(TEAMS[0], TEAMS[1])
    print(f"  {p['home']} {p['p_home']:.1%} / draw {p['p_draw']:.1%} / {p['away']} {p['p_away']:.1%}")

    table = current_table(m, div="E0", season="2026-27")
    fixtures = remaining_fixtures(m, list(table["team"]), "E0", "2026-27")
    print(f"\nsimulating {len(fixtures)} remaining fixtures")
    res = simulate_season(model, table, fixtures, n_sims=4000)
    print(format_summary(res, top=6))

    print("\nbacktest on 2025-26, all three targets:")
    print(f"  {'target':8}{'RPS':>9}{'logloss':>10}")
    for target in ("goals", "xg", "blend"):
        preds = walk_forward(m, test_season="2025-26", refit_every_days=21,
                             min_train_matches=300, verbose=False,
                             half_life_days=300, prior_sd=0.5, target=target)
        r = evaluate(preds)["model"]
        print(f"  {target:8}{r['rps']:>9.4f}{r['log_loss']:>10.4f}")

    print("\nbootstrap season intervals:")
    boots = bootstrap_fits(m, n_boot=15, verbose=False,
                           half_life_days=300, prior_sd=0.5, target="blend")
    wide = simulate_season(boots, table, fixtures, n_sims=2000)
    for label, r in (("point estimate", res), ("bootstrap", wide)):
        s_ = r["summary"]
        span = (s_["pts_90th_pct"] - s_["pts_10th_pct"]).mean()
        print(f"  {label:15} mean 10-90 points span {span:5.1f}, "
              f"top title {s_['p_title'].max():.1%}")
    assert (wide["summary"]["pts_90th_pct"] - wide["summary"]["pts_10th_pct"]).mean() > 0

    print("\nrating drift:")
    from eplmodel.drift import schedule_rounds
    rounds = schedule_rounds(fixtures, list(table["team"]))
    for r in rounds:
        seen = [t for pair in r for t in pair]
        assert len(seen) == len(set(seen)), "a team appears twice in one round"
    print(f"  {len(fixtures)} fixtures packed into {len(rounds)} rounds, no clashes")
    from eplmodel import rating_spread
    anchor = rating_spread(model, list(table["team"]))
    walk = simulate_season(boots, table, fixtures, n_sims=2000, drift_sigma=0.08)
    revert = simulate_season(boots, table, fixtures, n_sims=2000, drift_stationary_sd=anchor)
    print(f"  league rating spread {anchor:.3f}")
    for label, r in (("no drift", wide), ("random walk", walk), ("mean-reverting", revert)):
        s_ = r["summary"]
        span = (s_["pts_90th_pct"] - s_["pts_10th_pct"]).mean()
        tail = s_.tail(5)["p_title"].sum()
        print(f"  {label:15} span {span:5.1f}, top title {s_['p_title'].max():.1%}, "
              f"bottom-five title {tail:.2%}")
    # Mean reversion must keep the long-run spread bounded; a random walk does not.
    assert revert["drift_reversion"] > 0
    assert walk["drift_reversion"] == 0
    cal = calibrate_drift(m, ["2025-26"], early_week=4, mode="spread",
                          values=(0.0, anchor), n_boot=5, n_sims=800, verbose=False,
                          half_life_days=300, prior_sd=0.5, target="blend")
    assert 0.0 <= cal["coverage"] <= 1.0 and cal["mode"] == "spread"
    print(f"  calibration ran: picked spread {cal['value']:.3f}, "
          f"coverage {cal['coverage']:.0%}")

    print("\npost-hoc widening:")
    base_sim = simulate_season(boots, table, fixtures, n_sims=4000, seed=3)
    for k in (1.0, 1.3):
        r = simulate_season(boots, table, fixtures, n_sims=4000, widen=k, seed=3)
        s_ = r["summary"]
        span = (s_["pts_90th_pct"] - s_["pts_10th_pct"]).mean()
        print(f"  widen {k:.1f}: span {span:5.1f}, top title {s_['p_title'].max():.1%}, "
              f"bottom-five title {s_.tail(5)['p_title'].sum():.2%}")
    b = base_sim["summary"]
    w = simulate_season(boots, table, fixtures, n_sims=4000, widen=1.3, seed=3)["summary"]
    assert (w["pts_90th_pct"] - w["pts_10th_pct"]).mean() > (b["pts_90th_pct"] - b["pts_10th_pct"]).mean()
    assert abs(w["proj_points"].mean() - b["proj_points"].mean()) < 1.0, "widening must not move the centre"
    print("  widening stretches the spread and leaves the centre alone")

    print("\nmarket blending:")
    train = walk_forward(m, test_season="2024-25", refit_every_days=21,
                         min_train_matches=300, verbose=False,
                         half_life_days=300, prior_sd=0.5, target="blend")
    info = fit_weight(train)
    test = walk_forward(m, test_season="2025-26", refit_every_days=21,
                        min_train_matches=300, verbose=False,
                        half_life_days=300, prior_sd=0.5, target="blend")
    ev = evaluate_blend(test, info["weight"])
    print(f"  weight {info['weight']:.3f} on the model | "
          f"model {ev['model']['rps']:.4f}  market {ev['market']['rps']:.4f}  "
          f"blend {ev['blend']['rps']:.4f}")
    partial = test.head(6).copy()
    partial.loc[partial.index[:3], ["odds_h", "odds_d", "odds_a"]] = np.nan
    ab = apply_blend(partial, info["weight"])
    assert np.allclose(ab.loc[~ab["blended"], "b_home"], ab.loc[~ab["blended"], "m_home"])
    assert np.allclose(ab[["b_home", "b_draw", "b_away"]].sum(axis=1), 1.0)
    print("  fixtures without odds pass through unblended")

    print("\nxG helpers:")
    from eplmodel.xg import _find_column, reconcile_names
    probe = pd.DataFrame(columns=["date", "home_team", "home_xg", "away_xg", "away_team"])
    assert _find_column(probe, "home_xg", "xg_home") == "home_xg"
    known = {"Manchester United", "Tottenham Hotspur", "AFC Bournemouth"}
    got = reconcile_names(pd.DataFrame({"home_raw": ["Manchester Utd", "Tottenham"],
                                        "away_raw": ["Bournemouth", "Bournemouth"]}), known, verbose=False)
    assert list(got["home"]) == ["Manchester United", "Tottenham Hotspur"]
    assert list(got["away"]) == ["AFC Bournemouth", "AFC Bournemouth"]
    print("  column resolver and club-name reconciliation ok")

    print("\nall good.")


if __name__ == "__main__":
    main()