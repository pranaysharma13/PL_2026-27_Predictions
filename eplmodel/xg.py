"""Expected goals, fetched from FBref or Understat via `soccerdata`.

    pip install soccerdata

Two things about this module are worth knowing before you run it.

**It scrapes.** FBref throttles hard, so a first pull of several seasons runs
for minutes, not seconds. Everything is cached under `--xg-dir` and completed
seasons are never re-fetched. Scrapers also break when a site changes its HTML;
if that happens, `soccerdata` is the thing to upgrade.

**Coverage differs by source.** Understat covers the big five leagues only, so
it has no Championship data at all, which is exactly where Hull, Ipswich and
Coventry's history lives. FBref covers the Championship but needs a custom
league registered, which `_prepare()` below does. Use FBref unless you only
want the Premier League.

Never mix sources in one model. Providers' xG models differ enough that a
blended rating would be measuring two different things.
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import canonical, season_code, season_label

DIV_TO_LEAGUE = {"E0": "ENG-Premier League", "E1": "ENG-Championship"}
LEAGUE_TO_DIV = {v: k for k, v in DIV_TO_LEAGUE.items()}

# soccerdata only ships the Premier League for England. The Championship has to
# be registered before soccerdata is imported, because the custom league file is
# merged at import time.
CUSTOM_LEAGUES = {
    "ENG-Championship": {
        "MatchHistory": "E1",
        "FBref": "Championship",
        "ESPN": "eng.2",
        "WhoScored": "England - Championship",
        "season_start": "Aug",
        "season_end": "May",
    }
}

SOURCES_WITHOUT_CHAMPIONSHIP = {"understat"}


def _prepare(xg_dir: str):
    """Point soccerdata at a local directory and register extra leagues."""
    base = Path(xg_dir).expanduser().resolve()
    (base / "config").mkdir(parents=True, exist_ok=True)
    (base / "data").mkdir(parents=True, exist_ok=True)
    os.environ["SOCCERDATA_DIR"] = str(base)

    path = base / "config" / "league_dict.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    merged = {**existing, **CUSTOM_LEAGUES}
    if merged != existing:
        path.write_text(json.dumps(merged, indent=2))

    try:
        import soccerdata  # noqa: F401  (imported for its import-time config load)
    except ImportError as exc:
        raise ImportError("soccerdata is not installed. Run: pip install soccerdata") from exc
    import soccerdata as sd

    return sd


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Resolve a column by name, case and separator insensitively.

    Scraper column names drift between versions, so match loosely and fail with
    a readable message rather than a KeyError three functions later.
    """
    flat = {}
    for col in df.columns:
        key = "_".join(str(p) for p in col) if isinstance(col, tuple) else str(col)
        flat[key.lower().replace(" ", "_").replace("-", "_")] = col
    for cand in candidates:
        c = cand.lower().replace(" ", "_").replace("-", "_")
        if c in flat:
            return flat[c]
    for cand in candidates:
        c = cand.lower().replace(" ", "_").replace("-", "_")
        for key, col in flat.items():
            if key.endswith(c) or key.startswith(c):
                return col
    return None


def fetch_xg(
    start_year: int,
    end_year: int,
    divisions: tuple[str, ...] = ("E0",),
    source: str = "fbref",
    xg_dir: str = "xg_cache",
    verbose: bool = True,
) -> pd.DataFrame:
    """Pull match-level xG. Returns season, div, date, home, away, xg_h, xg_a."""
    source = source.lower()
    if source not in {"fbref", "understat"}:
        raise ValueError("source must be 'fbref' or 'understat'")

    divisions = tuple(divisions)
    if source in SOURCES_WITHOUT_CHAMPIONSHIP and "E1" in divisions:
        if verbose:
            print(f"note: {source} has no Championship coverage, skipping E1")
        divisions = tuple(d for d in divisions if d != "E1")
    if not divisions:
        raise ValueError("no divisions left to fetch")

    sd = _prepare(xg_dir)
    scraper_cls = sd.FBref if source == "fbref" else sd.Understat
    # "2019-2020" rather than "1920": the short form is ambiguous and soccerdata warns about it
    seasons = [f"{y}-{y + 1}" for y in range(start_year, end_year + 1)]

    frames = []
    for div in divisions:
        league = DIV_TO_LEAGUE[div]
        if verbose:
            print(f"fetching {source} xG for {league}, seasons {seasons[0]} to {seasons[-1]}")
            if source == "fbref":
                print("  (FBref drives a browser and rate-limits; the first run takes minutes)")
        scraper = scraper_cls(leagues=league, seasons=seasons)
        raw = scraper.read_schedule().reset_index()

        c_home = _find_column(raw, "home_team", "home")
        c_away = _find_column(raw, "away_team", "away")
        c_xgh = _find_column(raw, "home_xg", "xg_home", "home_expected_goals")
        c_xga = _find_column(raw, "away_xg", "xg_away", "away_expected_goals")
        c_date = _find_column(raw, "date", "game_date")
        c_season = _find_column(raw, "season")
        missing = [n for n, c in [("home", c_home), ("away", c_away),
                                  ("home xg", c_xgh), ("away xg", c_xga)] if c is None]
        if missing:
            cache = Path(xg_dir) / "data"
            raise RuntimeError(
                f"could not find {missing} in the {source} schedule.\n"
                f"available columns: {list(raw.columns)}\n"
                f"rows returned: {len(raw)}\n\n"
                "Most likely a poisoned cache. An interrupted or blocked scrape saves the\n"
                "page it got, and a block page still parses into a table with the right\n"
                "shape but no xG columns. Clear it and retry:\n"
                f"    rm -rf {cache}\n\n"
                "If the retry fails the same way in seconds rather than minutes, the site\n"
                "is refusing the scrape. Use --xg-source understat instead, or upgrade\n"
                "soccerdata in case its schema changed."
            )

        tidy = pd.DataFrame(
            {
                "div": div,
                "home_raw": raw[c_home].astype(str),
                "away_raw": raw[c_away].astype(str),
                "xg_h": pd.to_numeric(raw[c_xgh], errors="coerce"),
                "xg_a": pd.to_numeric(raw[c_xga], errors="coerce"),
                "date": pd.to_datetime(raw[c_date], errors="coerce") if c_date else pd.NaT,
                "season_raw": raw[c_season].astype(str) if c_season else "",
            }
        )
        tidy = tidy.dropna(subset=["xg_h", "xg_a"])
        tidy["season"] = tidy["season_raw"].map(_season_from_code)
        frames.append(tidy)

    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  {len(out)} matches with xG across {out['season'].nunique()} seasons")
    return out


def _season_from_code(code: str) -> str:
    """soccerdata's '2627' back to this project's '2026-27'."""
    code = str(code).strip()
    if len(code) == 4 and code.isdigit():
        yy = int(code[:2])
        return season_label(2000 + yy if yy < 90 else 1900 + yy)
    return code


def reconcile_names(xg: pd.DataFrame, known: set[str], verbose: bool = True) -> pd.DataFrame:
    """Map scraper club names onto the names used by football-data.co.uk.

    Exact alias match first, then a conservative fuzzy fallback. Every fuzzy
    match is printed, because a silently wrong one corrupts a team's ratings
    without ever raising an error.
    """
    xg = xg.copy()
    cache: dict[str, str | None] = {}

    def resolve(name: str) -> str | None:
        if name in cache:
            return cache[name]
        c = canonical(name)
        if c in known:
            cache[name] = c
            return c
        hit = difflib.get_close_matches(c, sorted(known), n=1, cutoff=0.80)
        if hit:
            if verbose:
                print(f"  fuzzy match: {name!r} -> {hit[0]!r}")
            cache[name] = hit[0]
            return hit[0]
        cache[name] = None
        return None

    xg["home"] = xg["home_raw"].map(resolve)
    xg["away"] = xg["away_raw"].map(resolve)

    unmatched = sorted(
        set(xg.loc[xg["home"].isna(), "home_raw"]) | set(xg.loc[xg["away"].isna(), "away_raw"])
    )
    if unmatched and verbose:
        print(f"  {len(unmatched)} club names could not be matched and will be dropped:")
        for n in unmatched:
            print(f"    {n!r}")
        print("  add them to TEAM_ALIASES in eplmodel/data.py")
    return xg.dropna(subset=["home", "away"])


def merge_xg(matches: pd.DataFrame, xg: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Attach xG to the match frame.

    Joined on (season, div, home, away) rather than on date. A given pairing at
    a given venue happens exactly once per season, and sources disagree about
    kickoff dates across timezones often enough that a date join silently loses
    matches.
    """
    known = set(matches["home"]) | set(matches["away"])
    xg = reconcile_names(xg, known, verbose=verbose)

    key = ["season", "div", "home", "away"]
    lookup = (
        xg.dropna(subset=["season"])
        .drop_duplicates(subset=key, keep="last")
        .set_index(key)[["xg_h", "xg_a"]]
    )

    out = matches.drop(columns=["xg_h", "xg_a"], errors="ignore").join(
        lookup, on=key, how="left"
    )

    n_total = len(out)
    n_hit = int(out["xg_h"].notna().sum())
    if verbose:
        print(f"  xG attached to {n_hit}/{n_total} matches ({n_hit / max(n_total, 1):.0%})")
        per_season = (
            out.assign(has=out["xg_h"].notna())
            .groupby(["season", "div"])["has"]
            .agg(["sum", "size"])
        )
        for (season, div), row in per_season.iterrows():
            flag = "" if row["sum"] else "   <- no coverage"
            print(f"    {season} {div}: {int(row['sum'])}/{int(row['size'])}{flag}")
        if n_hit == 0:
            print("  nothing matched. Check club-name reconciliation above.")
    return out


def sanity_check(matches: pd.DataFrame) -> pd.DataFrame:
    """Compare xG totals against actual goals. A useful smell test.

    Across a full season a team's xG and goals should be close. Big persistent
    gaps are either genuine finishing quality or a broken join; it is worth
    knowing which before you trust the ratings.
    """
    sub = matches.dropna(subset=["xg_h", "xg_a"])
    if sub.empty:
        return pd.DataFrame()
    home = sub.groupby("home").agg(gf=("hg", "sum"), xgf=("xg_h", "sum"),
                                   ga=("ag", "sum"), xga=("xg_a", "sum"))
    away = sub.groupby("away").agg(gf=("ag", "sum"), xgf=("xg_a", "sum"),
                                   ga=("hg", "sum"), xga=("xg_h", "sum"))
    tot = home.add(away, fill_value=0)
    tot["goals_minus_xg"] = tot["gf"] - tot["xgf"]
    tot["conceded_minus_xg"] = tot["ga"] - tot["xga"]
    tot.index.name = "team"
    return tot.sort_values("goals_minus_xg", ascending=False).round(1)