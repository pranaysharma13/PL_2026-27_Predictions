"""Data loading for the Premier League prediction model.

Source: football-data.co.uk season CSVs. One HTTP GET per season per division.
E0 = Premier League, E1 = Championship.

The schema this module produces is deliberately small and stable:

    date, div, season, home, away, hg, ag, [odds_h, odds_d, odds_a], [xg_h, xg_a]

`xg_h` / `xg_a` are always present but may be NaN. Nothing downstream requires
them, so you can bolt on an xG source later without touching the model code.
"""

from __future__ import annotations

import io
import os
import ssl
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# football-data.co.uk writes club names its own way, and occasionally changes
# them. Map everything onto one canonical set so joins never silently fail.
TEAM_ALIASES = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Notts Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "West Brom": "West Bromwich Albion",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Ipswich": "Ipswich Town",
    "Hull": "Hull City",
    "Coventry": "Coventry City",
    "Stoke": "Stoke City",
    "Cardiff": "Cardiff City",
    "Swansea": "Swansea City",
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Bolton": "Bolton Wanderers",
    "Derby": "Derby County",
    "Huddersfield": "Huddersfield Town",
    "Luton": "Luton Town",
    "Middlesbrough": "Middlesbrough",
    "Preston": "Preston North End",
    "QPR": "Queens Park Rangers",
    "Sheffield Weds": "Sheffield Wednesday",
    "Sheff Wed": "Sheffield Wednesday",
    "Bournemouth": "AFC Bournemouth",
    "Plymouth": "Plymouth Argyle",
    "Bristol City": "Bristol City",
    "Charlton": "Charlton Athletic",
    "Oxford": "Oxford United",
    "Portsmouth": "Portsmouth",
    "Sunderland": "Sunderland",
    "Millwall": "Millwall",
    "Watford": "Watford",
    "Burnley": "Burnley",
    "Blackpool": "Blackpool",
    "Wigan": "Wigan Athletic",
    "Wycombe": "Wycombe Wanderers",
    "Rotherham": "Rotherham United",
    # FBref spellings
    "Manchester Utd": "Manchester United",
    "Newcastle Utd": "Newcastle United",
    "Nott'ham Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Peterborough Utd": "Peterborough United",
    "Bristol Rvs": "Bristol Rovers",
    # Understat spellings
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Nottingham Forest": "Nottingham Forest",
    "Leicester": "Leicester City",
}

# Which bookmaker columns to use, in order of preference. The distinction
# matters more than it looks.
#
# Closing odds are the sharpest public estimate in existence, incorporating
# everything known up to kickoff including confirmed team sheets. That is the
# benchmark for "is my model as good as the market", and it is brutal.
#
# Opening odds are what you could actually act on days ahead, before lineups
# are known. That is the benchmark for "does my model know anything useful
# early". A model can be hopeless against the close and valuable against the
# open, and the gap between the two is a measure of how much of the market's
# edge is late information rather than better modelling.
CLOSING_PREFERENCES = [
    ("AvgCH", "AvgCD", "AvgCA"),     # market average, closing
    ("B365CH", "B365CD", "B365CA"),  # Bet365, closing
    ("AvgH", "AvgD", "AvgA"),        # fall back to opening if no closing line
    ("BbAvH", "BbAvD", "BbAvA"),
    ("B365H", "B365D", "B365A"),
]

OPENING_PREFERENCES = [
    ("AvgH", "AvgD", "AvgA"),        # market average, opening
    ("B365H", "B365D", "B365A"),     # Bet365, opening
    ("BbAvH", "BbAvD", "BbAvA"),     # legacy Betbrain average
    ("AvgCH", "AvgCD", "AvgCA"),     # fall back to closing if no opening line
    ("B365CH", "B365CD", "B365CA"),
]

ODDS_PREFERENCES = CLOSING_PREFERENCES  # kept for backwards compatibility


def _preferences(odds: str):
    if odds == "closing":
        return CLOSING_PREFERENCES
    if odds == "opening":
        return OPENING_PREFERENCES
    raise ValueError(f"odds must be 'closing' or 'opening', got {odds!r}")


def season_code(start_year: int) -> str:
    """2026 -> '2627' (the 2026-27 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def canonical(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


@dataclass
class LoadReport:
    ok: list[tuple[int, str]]
    failed: list[tuple[int, str, str]]

    def summary(self) -> str:
        lines = [f"loaded {len(self.ok)} season-division files"]
        for year, div, err in self.failed:
            lines.append(f"  ! {season_label(year)} {div}: {err}")
        return "\n".join(lines)


def _ssl_context() -> ssl.SSLContext:
    """A context with a working CA bundle.

    Python installed from python.org on macOS ships an empty certificate store,
    so plain urllib raises CERTIFICATE_VERIFY_FAILED on every HTTPS request.
    certifi carries the Mozilla CA bundle and is almost always already present,
    since requests depends on it.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


def _download(year: int, div: str, cache_dir: str | None) -> bytes:
    url = f"{BASE_URL}/{season_code(year)}/{div}.csv"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{season_code(year)}_{div}.csv")
        # Never cache the in-progress season: it changes twice a week.
        if os.path.exists(path) and _is_completed_season(path):
            with open(path, "rb") as fh:
                return fh.read()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as resp:
        raw = resp.read()
    if cache_dir:
        with open(os.path.join(cache_dir, f"{season_code(year)}_{div}.csv"), "wb") as fh:
            fh.write(raw)
    return raw


def _is_completed_season(path: str) -> bool:
    """A 20-team division plays 380 matches; 24 teams play 552."""
    try:
        with open(path, "rb") as fh:
            rows = sum(1 for line in fh if line.strip())
        return rows >= 380
    except OSError:
        return False


def _strip_bom(df: pd.DataFrame) -> pd.DataFrame:
    """Remove a byte-order mark glued to the first column name.

    football-data.co.uk writes some files as UTF-8 with a BOM. Read as
    latin-1, those three bytes survive as a visible prefix, turning "Div" into
    "\u00ef\u00bb\u00bfDiv". Reading as utf-8-sig would fix the BOM but break on club
    names with accented characters, so strip it after the fact instead.
    """
    df.columns = [str(c).replace("\ufeff", "").replace("\u00ef\u00bb\u00bf", "").strip() for c in df.columns]
    return df


def _pick_odds(df: pd.DataFrame, odds: str = "closing") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=["odds_h", "odds_d", "odds_a"], dtype="float64")
    for h, d, a in _preferences(odds):
        if h in df.columns and d in df.columns and a in df.columns:
            block = df[[h, d, a]].apply(pd.to_numeric, errors="coerce")
            fill = out["odds_h"].isna() & block[h].notna()
            out.loc[fill, "odds_h"] = block.loc[fill, h]
            out.loc[fill, "odds_d"] = block.loc[fill, d]
            out.loc[fill, "odds_a"] = block.loc[fill, a]
    return out


def load_matches(
    start_year: int,
    end_year: int,
    divisions: tuple[str, ...] = ("E0",),
    cache_dir: str | None = "data_cache",
    odds: str = "closing",
    verbose: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load every completed match for the given seasons and divisions.

    `start_year` and `end_year` are season start years, inclusive. Passing
    2021, 2026 gives you 2021-22 through 2026-27.
    """
    frames, ok, failed = [], [], []
    for year in range(start_year, end_year + 1):
        for div in divisions:
            try:
                raw = _download(year, div, cache_dir)
            except Exception as exc:  # network, 404 on a not-yet-published season
                failed.append((year, div, f"{type(exc).__name__}: {exc}"))
                continue
            try:
                df = _strip_bom(pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip"))
            except Exception as exc:
                failed.append((year, div, f"parse error: {exc}"))
                continue
            needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
            if not needed.issubset(df.columns):
                failed.append((year, div, f"missing columns: {sorted(needed - set(df.columns))}"))
                continue

            tidy = pd.DataFrame(
                {
                    "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
                    "div": div,
                    "season": season_label(year),
                    "season_start": year,
                    "home": df["HomeTeam"].map(canonical),
                    "away": df["AwayTeam"].map(canonical),
                    "hg": pd.to_numeric(df["FTHG"], errors="coerce"),
                    "ag": pd.to_numeric(df["FTAG"], errors="coerce"),
                }
            )
            tidy = pd.concat([tidy, _pick_odds(df, odds)], axis=1)
            tidy["xg_h"] = np.nan
            tidy["xg_a"] = np.nan
            tidy = tidy.dropna(subset=["date", "home", "away", "hg", "ag"])
            tidy["hg"] = tidy["hg"].astype(int)
            tidy["ag"] = tidy["ag"].astype(int)
            frames.append(tidy)
            ok.append((year, div))

    report = LoadReport(ok=ok, failed=failed)
    if not frames:
        raise RuntimeError(
            "No data loaded.\n"
            + report.summary()
            + "\n\nIf every season failed, football-data.co.uk is probably "
            "unreachable from this machine. You can download the CSVs by hand "
            f"from {BASE_URL}/<season>/E0.csv and point --cache-dir at them."
        )
    matches = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if verbose:
        print(report.summary())
        print(f"{len(matches)} matches, {matches['date'].min().date()} to "
              f"{matches['date'].max().date()}, using {odds} odds")
    return matches, report


FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def load_upcoming_fixtures(divisions: tuple[str, ...] = ("E0",), odds: str = "closing",
                           verbose: bool = True) -> pd.DataFrame:
    """Next week's fixtures with bookmaker odds.

    football-data.co.uk publishes a single rolling file covering the coming
    week across all its leagues. This is the only place odds for unplayed
    matches come from, which is why market blending can sharpen this weekend's
    predictions but can do nothing for a fixture in April.
    """
    req = urllib.request.Request(FIXTURES_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as resp:
        raw = resp.read()
    df = _strip_bom(pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip"))

    needed = {"Div", "Date", "HomeTeam", "AwayTeam"}
    if not needed.issubset(df.columns):
        raise RuntimeError(
            f"unexpected fixtures file; missing {sorted(needed - set(df.columns))}\n"
            f"columns found: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "div": df["Div"].astype(str),
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "home": df["HomeTeam"].map(canonical),
            "away": df["AwayTeam"].map(canonical),
        }
    )
    out = pd.concat([out, _pick_odds(df, odds)], axis=1)
    out = out[out["div"].isin(divisions)].dropna(subset=["home", "away"])
    out = out.sort_values("date").reset_index(drop=True)
    if verbose:
        n_odds = int(out[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1).sum())
        print(f"{len(out)} upcoming fixtures, {n_odds} with odds")
    return out


def load_custom(path: str) -> pd.DataFrame:
    """Load your own CSV instead. Required columns: date, home, away, hg, ag.

    Optional: div, season, odds_h, odds_d, odds_a, xg_h, xg_a.
    """
    df = pd.read_csv(path)
    missing = {"date", "home", "away", "hg", "ag"} - set(df.columns)
    if missing:
        raise ValueError(f"custom CSV is missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home"] = df["home"].map(canonical)
    df["away"] = df["away"].map(canonical)
    for col, default in [("div", "E0"), ("season", "custom"), ("season_start", 0)]:
        if col not in df.columns:
            df[col] = default
    for col in ["odds_h", "odds_d", "odds_a", "xg_h", "xg_a"]:
        if col not in df.columns:
            df[col] = np.nan
    return df.dropna(subset=["date", "home", "away", "hg", "ag"]).sort_values("date").reset_index(drop=True)


def current_table(matches: pd.DataFrame, div: str = "E0", season: str | None = None) -> pd.DataFrame:
    """League table from played matches. 3 points a win, PL tiebreakers."""
    sub = matches[matches["div"] == div]
    if season is not None:
        sub = sub[sub["season"] == season]
    teams = sorted(set(sub["home"]) | set(sub["away"]))
    rows = {t: dict(team=t, played=0, won=0, drawn=0, lost=0, gf=0, ga=0, points=0) for t in teams}
    for r in sub.itertuples(index=False):
        h, a = rows[r.home], rows[r.away]
        h["played"] += 1
        a["played"] += 1
        h["gf"] += r.hg
        h["ga"] += r.ag
        a["gf"] += r.ag
        a["ga"] += r.hg
        if r.hg > r.ag:
            h["won"] += 1
            a["lost"] += 1
            h["points"] += 3
        elif r.hg < r.ag:
            a["won"] += 1
            h["lost"] += 1
            a["points"] += 3
        else:
            h["drawn"] += 1
            a["drawn"] += 1
            h["points"] += 1
            a["points"] += 1
    table = pd.DataFrame(rows.values())
    table["gd"] = table["gf"] - table["ga"]
    table = table.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)
    table.index += 1
    return table[["team", "played", "won", "drawn", "lost", "gf", "ga", "gd", "points"]]


def remaining_fixtures(matches: pd.DataFrame, teams: list[str], div: str, season: str) -> list[tuple[str, str]]:
    """Every (home, away) pair in a double round robin that hasn't been played.

    This is why you don't need a fixture-list feed: a league season is a
    complete double round robin, so the remaining fixtures are just the full
    set minus what's already happened. Dates don't matter for simulation.
    """
    played = {
        (r.home, r.away)
        for r in matches[(matches["div"] == div) & (matches["season"] == season)].itertuples(index=False)
    }
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]