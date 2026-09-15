# Premier League 2026-27 prediction model

A Dixon-Coles goal model plus a Monte Carlo season simulator. Fits team attack
and defence ratings from historical results, turns those into probabilities for
any fixture, and simulates the remaining fixtures to get title, top-four and
relegation odds.

## Setup

One-off, the first time only:

```bash
cd PL_2026-27_Predictions
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas scipy soccerdata
python smoke_test.py               # offline check, no network needed
```

## Every session after that

The virtual environment has to be activated in each new terminal. Nothing else
works until it is, and the error you get if you forget is a confusing
`ModuleNotFoundError` for numpy rather than anything about environments.

```bash
cd PL_2026-27_Predictions
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Your prompt turns into `(.venv) ...` when it's active. That's the signal.
Activation does not carry across terminal tabs or windows, so each new one
needs the command again. Running it twice is harmless. `deactivate` exits, and
so does closing the terminal.

VS Code often activates it automatically in the integrated terminal if you've
selected the `.venv` interpreter (Cmd+Shift+P, "Python: Select Interpreter").
Check the prompt before typing.

Then, with network access:

```bash
python run.py fetch-xg                                  # scrape xG once, cached after
python run.py table                                     # current league table
python run.py ratings                                   # fitted team strengths
python run.py predict --home Arsenal --away "Leeds United"
python run.py simulate --sims 20000 --out projections.csv
python run.py backtest --test-season 2025-26 --compare-targets
python run.py blend --test-season 2025-26        # fit and score a market blend
python run.py card                               # price the upcoming round
python run.py tune --test-season 2025-26
```

Global options go **before** the subcommand:

```bash
python run.py --half-life 180 --prior-sd 0.3 simulate --sims 50000
```

## Where the data comes from

**Results and odds**: season CSVs from football-data.co.uk, one HTTP GET per
season per division, cached in `data_cache/`. The current season is never
cached, since it updates twice a week.

**Expected goals**: Understat by default, scraped through `soccerdata` and
cached in `xg_cache/`. Run `fetch-xg` once; everything afterwards reads the
cache, which is keyed by source.

    python run.py fetch-xg                    # Understat, Premier League only
    python run.py --xg-source fbref fetch-xg  # see the warning below
    python run.py fetch-xg --refresh          # re-scrape

Understat embeds its data as JSON in the page, so a plain HTTP request gets
everything. No browser, no rate limiting, done in under a minute. It computes
its own xG from a neural network trained on shot-level data. The limitation is
coverage: big five leagues only, so no Championship. `fetch-xg` says so and
skips E1.

FBref publishes StatsBomb xG and does cover the Championship, but it is the
harder path and **as of September 2026 it does not work**. Its scraper drives a
real Chrome browser through `seleniumbase`, which downloads a patched
ChromeDriver on first use and can trip macOS App Management protection. Worse,
FBref currently returns a Scores & Fixtures table with the xG columns absent
while every other column arrives intact, which suggests either a markup change
`soccerdata` hasn't caught up with or a stripped-down table served to suspected
bots. `soccerdata` 1.9.1 is the latest release, so there is no upgrade to reach
for. If you need Championship xG, FBref's per-team match logs
(`read_team_match_stats`) are a different endpoint that may still carry it.

`soccerdata` only ships `ENG-Premier League`, so the Championship is registered
as a custom league automatically in `xg_cache/config/league_dict.json`.

Don't mix sources in one model. Providers' xG models differ enough that blended
ratings would be measuring two different things.

Missing Championship xG costs less than it sounds like. Those matches still
load from football-data.co.uk, so promoted clubs keep their full second-tier
results history and still get properly estimated ratings. Those rows just
contribute through the Poisson term rather than the Gamma one, which the mixed
likelihood handles by design.

xG is joined on `(season, division, home, away)` rather than on date, because a
given pairing at a given venue happens exactly once per season and sources
disagree about kickoff dates across timezones. Club names are reconciled through
the alias table with a conservative fuzzy fallback; every fuzzy match and every
failure is printed, since a silently wrong match corrupts a team's ratings
without raising an error. `fetch-xg` also prints the biggest goals-minus-xG gaps,
which doubles as a check that the join actually worked.

Team names are normalised through `TEAM_ALIASES` in `eplmodel/data.py`. If a
club appears twice under two spellings, add it there.

To use your own data instead:

```bash
python run.py --csv my_matches.csv simulate
```

Required columns: `date, home, away, hg, ag`. Optional: `div, season, odds_h,
odds_d, odds_a, xg_h, xg_a`. If your CSV already has xG, nothing is scraped.

## How the model works

```
lambda_home = exp(mu_div + home_adv + attack_home - defence_away)
lambda_away = exp(mu_div           + attack_away - defence_home)
```

Match probabilities always come from a Dixon-Coles score matrix on those rates:
Poisson marginals plus the `tau` correction for 0-0, 1-0, 0-1 and 1-1, which
independent Poisson systematically gets wrong. Fitted by weighted maximum
likelihood with an analytic gradient, so a full fit takes well under a second.

### What the ratings are fitted on

`--target` chooses the estimation target. Match probabilities are unaffected in
form; only the rates change.

| target | likelihood | on |
|---|---|---|
| `goals` | Poisson + tau | actual goals |
| `xg` | Gamma | expected goals |
| `blend` | Gamma | `w * xG + (1-w) * goals`, default `w = 0.7` |

`blend` is the default. xG is a much less noisy measurement of the same
underlying rate, but it discards finishing skill, which is partly real, so a
mixture usually beats either alone. Matches with no xG fall back to the Poisson
term automatically, so patchy coverage across seasons and divisions is fine in
a single fit. The Gamma shape is estimated rather than asserted, so how much
more an xG match counts than a goals-only match comes out of the data.

Because the tau correction describes the discrete scoreline, `rho` is
re-estimated from actual goals after the rates are fitted on xG.

On synthetic data where the truth is known, fitting on xG roughly halves the
error in recovered team strength over a full season. In the regime that matters
here, one prior season plus four matchweeks with squads changed over the summer,
correlation with true current strength went from 0.80 (goals) to 0.89 (xG or
blend), with lower run-to-run variance. Check it on real data with
`backtest --compare-targets`.

### Three additions that matter for a season only four matchweeks old

**Time decay.** Every match is weighted `exp(-ln2 * age_days / half_life)`.
Default half-life 240 days. Shorter reacts faster to a squad rebuild and is
noisier; longer is more stable and slower to notice that a team has changed.

**Shrinkage.** A Gaussian prior pulls attack and defence toward the league
average, strength set by `--prior-sd`. This is what stops Coventry on zero
points from four games being handed a catastrophic rating, and Arsenal on
twelve from being handed an invincible one. Smaller value, harder shrinkage.

**Joint fit across divisions.** By default it fits the Premier League and the
Championship together. Team strength belongs to the team, not the league, so
clubs that have recently moved between the two divisions bridge them, which
gives Hull, Ipswich and Coventry real ratings derived from their Championship
form rather than a guess. Turn it off with `--no-championship`.

## Market blending

The closing line is hard to beat because it aggregates everyone else's model
plus information yours cannot see: injuries, lineups, transfer news, a manager
about to be sacked. Your model sees seven seasons of shot-level performance in a
consistent way no individual bettor does. Neither dominates, so a weighted
combination usually beats both.

    python run.py blend --fit-seasons 2023-24,2024-25 --test-season 2025-26

The weight is fitted, never chosen, and fitted on seasons you do not report on.
`--pooling log` (the default) takes a weighted geometric mean, which sharpens
when both sources agree; `--pooling linear` is the plain weighted average and
is more conservative. The fitted weight is as informative as the accuracy gain:
near 0.5 means your model contributes real independent signal, near 0.1 means
it is mostly redundant beside the market.

If the blend fails to beat the market on a held-out season, the output says so
outright rather than burying it.

**Measured result on 2025-26, recorded so it is not re-derived**: the fitted
weight is 0.000 against both closing and opening odds. The model adds nothing
to either. Opening odds score 0.2057 and closing 0.2048, so the market improves
by only 0.0009 across the week before kickoff; its advantage is not late team
news, it is simply better from the moment the line opens. Match-level prediction
against the market looks closed off. The season projections are where this model
has something the market does not price.

**This cannot help season projections.** Blending needs odds for the fixture
being priced, and nobody prices a match in April. football-data.co.uk publishes
a rolling file covering the coming week only, which is what `card` reads. So
one model serves both uses: `apply_blend` blends wherever odds exist and passes
everything else through untouched, so this weekend's fixtures get the sharper
number and the remaining 330 come through as the raw model, which is exactly
what the simulator needs.

## Uncertainty in the projections

Three different things are uncertain about a projected league table.

**Match randomness**: a better team loses sometimes. Sampling scorelines
captures this, and it is all the simulator does by default.

**Rating uncertainty**: you don't know exactly how good each team is *today*,
having estimated it from finitely many matches. `--bootstrap N` refits the
model on N resamples and spreads the simulations across them. Costs seconds.

**Rating drift**: teams change. Injuries, form, a sacking, January. A rating
fitted in September describes today well and March badly. Letting ratings
wander during the simulation makes uncertainty about a match grow with how far
away it is.

    python run.py drift                                            # calibrate
    python run.py simulate --sims 20000 --bootstrap 40 --drift-spread 0.30
    python run.py simulate --sims 20000 --bootstrap 40 --drift-auto

Ignoring the last two makes September projections confidently wrong, which is
when people most want to read them.

### Why drift is mean-reverting

A pure random walk has unbounded variance. Over 34 matchweeks that lets a
bottom club drift all the way to title-winning, which real teams do not do.
Measured on the real 2026-27 table, an unbounded walk gave clubs sitting on one
point from four matches a 1.4% title chance, which is not a defensible number.

So drift is mean-reverting instead: ratings wander but are pulled back toward
the fitted value, and their spread grows and then settles rather than expanding
forever. `--drift-spread` is that settling point, and it is the knob worth
using. `--drift-half-life` sets how fast it gets there, default 15 matchweeks.
`--drift SIGMA` still gives the pure random walk if you want to compare, but it
is not recommended.

The settling point is what gets calibrated, not the step size, for two reasons.
It is the quantity end-of-season coverage can actually identify, since many
combinations of step and reversion produce the same final spread. And it has a
natural scale: the spread of team strengths across the league. `drift` searches
multiples of that spread, so the answer reads as "a club's plausible range a
season out is about 1.5 times the range clubs currently occupy" rather than as
an opaque per-week number.

### Calibrating the drift

`drift` picks the value by asking whether past projections turned out
honest. For each completed season it stands at matchweek 4, projects the final
table, and checks where each club's real final points landed in the predicted
distribution. If the intervals are right, about 80% of clubs should land inside
their own 10th-to-90th percentile band. It reports coverage for each candidate and picks the
closest, warning you if the best value is the largest tested.

**Measured result on seven seasons, 140 club-seasons**: with no drift, coverage
was 68.6% against a target of 80%. Roughly one club in three finished outside a
band meant to hold four in five. Mean PIT sat between 0.495 and 0.506 at every
setting, so the central estimates were never biased; only the spread was wrong.

### What the under-coverage is not

Reaching 80% required a long-run spread of 0.470, which is three times the
spread of team strengths across the whole league. Taken literally that says a
club could plausibly end up three times further from its current rating than
the gap between the best and worst teams in the division, which is not a
description of football. At that setting reversion is too weak to bite within
one season, and clubs sitting on one point kept title chances above 1%. So
drift is not the mechanism.

Shrinkage was the other candidate, on the theory that compressed ratings give
compressed projected points. Halving it, from `prior_sd` 0.40 to 0.80, moved
coverage from 68.6% to 67.9%. Not that either.

By elimination the bootstrap is the prime suspect: resampling matches disturbs
the time-decay weighting the model depends on, and a bootstrap cannot see model
misspecification at all, only sampling variability given the model is right.
Untested. Regime shifts are the other candidate, since a November sacking is a
jump, not diffusion, which would explain why a smooth random walk needed an
absurd step size to approximate one.

### The honest correction

`--widen K` stretches each club's simulated points around its own mean before
the table is ranked. It changes nothing about the dynamics and claims no
mechanism. It exists because the projections are known to run about 15% narrow
for reasons that are not yet understood, and reporting corrected numbers with a
stated caveat beats reporting confident ones that are wrong.

    python run.py drift --mode widen          # calibrate the factor
    python run.py simulate --bootstrap 40 --widen 1.3

Because each club is stretched around its own mean, orderings and the
correlation between clubs survive, so title and relegation probabilities soften
rather than becoming incoherent. Some leakage to the bottom is unavoidable in
any widening: at 1.3 the bottom five clubs collectively pick up about 0.2% of
title probability. That is far less than drift produced at comparable width,
but it is not zero, and it is the cost of honest intervals.

The number it returns is a calibration knob, not a measurement. On synthetic
seasons with no drift at all it still settles on a small positive value,
because the bootstrap alone leaves projections slightly too narrow and this
parameter absorbs that too. Honest intervals are the goal, not a decomposition
of where the uncertainty came from.

**An approach that does not work**, recorded so nobody repeats it: fitting
ratings early and late in each season, treating the difference as drift, and
subtracting bootstrap variance to remove estimation error. Tested against
synthetic seasons with a known step size it fails in both directions, reporting
substantial drift where the truth is zero with a short measurement half-life,
and compressing real drift to a third of its size with a long one. No half-life
fixes it.

## Why no fixture list is needed

A league season is a complete double round robin. The remaining fixtures are
every ordered pair of teams minus the ones already played, which is computed
directly from results in `data.remaining_fixtures`. Dates are irrelevant to the
simulation, so no fixture feed is required.

## Reading the backtest

`backtest` walks forward through a season, refitting weekly on data strictly
before each block and scoring the predictions it makes. Never train on a season
and score it.

Rough calibration for the Premier League: RPS around 0.21 is a serious model,
around 0.19 is the bookmaker closing line. The `base rate` row is the
unconditional home/draw/away split. If the model doesn't beat that, something
is broken. Matching the closing line is the real bar, and it is hard, because
the line already contains everybody else's model.

The calibration table asks a different question: when the model says 30%, does
it happen 30% of the time? A model can rank matches well and still be
overconfident.

## What to add next, in order of payoff

1. **Availability and congestion.** Days of rest, European midweek fixtures,
   and the absence of a key player are all worth real goals. This is the
   largest remaining source of information the market has and the model does
   not. Note that the market's edge appears immediately at the open rather than
   accruing through the week, so this may be less of the story than it looks.
2. **Per-team home advantage.** Some grounds are worth more than others,
   though this needs a lot of data to estimate without overfitting.
3. **Shot-level data.** Understat's `read_shot_events` gives every shot with
   its xG, which opens up possession value, shot-quality splits and
   goalkeeper-adjusted defensive ratings.

## Honest caveats

- Four matchweeks is almost no information. Early-season projections are
  dominated by the prior, which is correct behaviour, not a bug.
- `tune` searches on one season and reports that season's score, which
  overstates performance. Tune on one season, report on another.
- The Understat path is confirmed working: 2,700 Premier League matches with xG
  across eight seasons. The FBref path is not, for the reasons above.
- If the column resolver fails, it prints the schema it actually received. A
  table with the right shape and no xG columns usually means the source refused
  the scrape rather than that the code is wrong.
- Beating the closing line consistently is genuinely difficult, and most models
  that appear to be doing so are leaking future information somewhere. If
  results look too good, look for the leak first.