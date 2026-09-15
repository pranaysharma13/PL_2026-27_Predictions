# Complete guide

Everything in this project: what it does, how to use it, what was built and
why, and what is and isn't trustworthy.

---

## 1. What this is, in one paragraph

A statistical model of Premier League football. It reads seven seasons of match
results and shot-quality data, works out how good every team is at scoring and
at preventing goals, and turns that into probabilities. Give it two teams and
it tells you how likely each outcome is. Ask it about the season and it plays
out all 340 remaining fixtures twenty thousand times and counts how often each
club wins the league or goes down.

---

## 2. Starting a session

Every command in this guide assumes the virtual environment is active. It has
to be activated in each new terminal, and if you forget, you get a
`ModuleNotFoundError` for numpy rather than anything that mentions
environments.

```bash
cd ~/Projects/PL_2026-27_Predictions
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Your prompt turns into `(.venv) ...` when it's active. It does not carry across
terminal tabs, so each new one needs the command again. Running it twice does
nothing bad. `deactivate` exits, and so does closing the terminal.

You only ever create the venv once. `python3 -m venv .venv` was a one-off and
the folder persists; activating just points `python` and `pip` at it for the
session. VS Code often does it automatically in the integrated terminal if
you've selected the `.venv` interpreter, so check the prompt first.

---

## 3. Answering "who wins the next match"

The short version:

```bash
python run.py predict --home Arsenal --away "Manchester City"
```

Output looks like this:

```
Arsenal vs Manchester City
  expected goals   1.91 - 1.77
  home win          41.7%   (fair odds 2.40)
  draw              22.4%   (fair odds 4.46)
  away win          35.9%   (fair odds 2.78)
  over 2.5 goals    71.1%
  both teams score  71.1%
  most likely scores: 1-1 (8.9%), 2-1 (8.1%), 1-2 (7.5%)
```

**The model never tells you who wins.** It tells you how likely each result is.
That distinction is the entire point. A 42% favourite loses more often than
not. If the model says 42% and you want a single name, the honest answer is
"Arsenal, but I'd expect to be wrong three times in five".

"Fair odds" is the decimal price at which a bet would break even. If a
bookmaker offers better than 2.40 on Arsenal, the model thinks there's value.
Given what the backtests found, the bookmaker is probably right and you are
probably wrong, but the number is there.

For the whole upcoming round at once:

```bash
python run.py card
```

That pulls this week's fixtures with their odds, prices each one, and shows the
model's numbers next to the market's. Use `--out card.csv` to save it.

Team names must match how football-data.co.uk writes them, after the alias
table maps them. Use `python run.py table` to see the exact spellings.

---

## 4. The files

```
PL_2026-27_Predictions/
├── run.py              command-line interface, all ten commands
├── smoke_test.py       offline self-test, no network needed
├── README.md           reference documentation
├── GUIDE.md            this file
├── data_cache/         downloaded results (created automatically)
├── xg_cache/           downloaded xG (created automatically)
└── eplmodel/
    ├── __init__.py     makes the folder a package, exports the public names
    ├── data.py         downloading, parsing, club-name matching, league tables
    ├── model.py        the model itself: fitting ratings, pricing matches
    ├── simulate.py     Monte Carlo of the rest of the season
    ├── backtest.py     walk-forward validation and scoring
    ├── xg.py           scraping expected goals
    ├── blend.py        combining model probabilities with bookmaker odds
    └── drift.py        calibrating how uncertain the projections should be
```

`run.py` sits outside `eplmodel/` because it's the thing you run, not part of
the library. Always run commands from the project root, since `data_cache/` and
`xg_cache/` are created in whatever directory you're standing in.

---

## 5. Every command

Global options go **before** the subcommand. Command options go after.

```bash
python run.py --half-life 180 simulate --sims 50000
#             ^^^^^^^^^^^^^^^ global    ^^^^^^^^^^^ command
```

### `table`
Current league standings, computed from downloaded results.

```bash
python run.py table
```
Use it to check the data loaded correctly and to see exact club spellings.

### `ratings`
Each club's fitted attack and defence strength.

```bash
python run.py ratings
```
Numbers are on a log scale where zero is average. Attack +0.30 means a club
scores about 35% more than average, since e^0.30 ≈ 1.35. Higher defence means
a better defence. `overall` is the two added together.

This is the model's actual opinion, stripped of fixtures and luck. A club can
sit high in the table and low here, which usually means they've been fortunate.

### `predict`
Probabilities for one fixture. Covered in section 3.

### `card`
The whole upcoming round, with model and market numbers side by side.

```bash
python run.py card --out card.csv
```
Fixtures already played are filtered out automatically. If the feed only
contains the round just finished, it says so rather than showing stale games.

### `simulate`
Projects the final table.

```bash
python run.py simulate --sims 20000 --bootstrap 40 --widen 1.5 --out projections.csv
```

| Option | What it does |
|---|---|
| `--sims N` | how many seasons to play out; 20,000 is plenty |
| `--bootstrap N` | refit the model N times on resampled data so the intervals account for not knowing ratings exactly |
| `--widen K` | stretch the reported spread by K; **use 1.5**, see section 8 |
| `--widen-auto` | work out K from past seasons first (slow) |
| `--drift-spread SD` | let ratings wander over the season; investigated and rejected, see section 8 |
| `--out FILE` | save to CSV |
| `--seed N` | change the random seed |

Output columns: `pts` current points, `proj` projected final points, `p10`/`p90`
the 10th and 90th percentile, then probabilities of winning the title,
finishing top four, top five, and being relegated.

### `backtest`
The honest test. Walks through a past season week by week, fits on only what
was known at the time, predicts the next round, and scores it against what
happened and against the bookmaker's closing odds.

```bash
python run.py backtest --test-season 2025-26 --compare-targets
```

`--compare-targets` runs all three fitting targets side by side. Read RPS,
lower is better. The `base rate` row is what you'd score by ignoring the teams
entirely; failing to beat it means something is broken.

### `blend`
Fits a weighted combination of your probabilities and the bookmaker's, then
scores it on a season the weight hasn't seen.

```bash
python run.py blend --fit-seasons 2023-24,2024-25 --test-season 2025-26
```

### `drift`
Checks whether past projections turned out honest, and finds the correction
that makes them so.

```bash
python run.py drift --mode widen
python run.py drift --mode widen --values 1.4,1.5,1.6,1.7
```

`--mode` picks what's being calibrated: `widen` (the post-hoc stretch, the one
that works), `spread` (mean-reverting drift), `sigma` (pure random walk). Takes
several minutes.

### `tune`
Grid search over the half-life and shrinkage settings.

```bash
python run.py tune --test-season 2024-25
```
Worth running once to see the surface is flat. Don't expect gains, see section 8.

### `fetch-xg`
Downloads expected-goals data. Run once; everything afterwards reads the cache.

```bash
python run.py fetch-xg
python run.py fetch-xg --refresh
```

### Global options

| Option | Default | What it does |
|---|---|---|
| `--target` | `blend` | what ratings are fitted on: `goals`, `xg`, or `blend` |
| `--blend-weight` | 0.70 | weight on xG when blending with goals |
| `--half-life` | 240 | days for a match's influence to halve |
| `--prior-sd` | 0.40 | shrinkage toward average; smaller pulls harder |
| `--from-season` | 2019 | earliest season to load |
| `--odds` | `closing` | `closing` or `opening` bookmaker prices |
| `--xg-source` | `understat` | `understat` or `fbref` |
| `--no-championship` | off | fit on the Premier League alone |
| `--csv FILE` | — | use your own data instead of downloading |

---

## 6. How the model works

### In plain terms

Every team gets two numbers: how good they are at scoring, and how good they
are at stopping goals. A match between two teams gives each side an expected
number of goals, worked out from one team's attack against the other's defence,
plus a bonus for playing at home. From those two expected values, the model
calculates the chance of every possible scoreline from 0-0 up to 12-12, adds up
all the ones where the home team scores more, and that's the home win
probability.

The team numbers come from fitting all seven seasons at once, finding the
values that best explain every result that actually happened. Recent matches
count more than old ones. Teams with few matches get pulled toward average, so
four good games doesn't make a promoted club look like a title contender.

### In technical terms

A Dixon-Coles bivariate Poisson model:

```
λ_home = exp(μ_div + home_adv + attack_home − defence_away)
λ_away = exp(μ_div            + attack_away − defence_home)
```

Goals are Poisson around those rates, with the Dixon-Coles τ correction applied
to 0-0, 1-0, 0-1 and 1-1, which independent Poisson systematically misprices.
Fitted by weighted maximum likelihood with an analytic gradient, so a full fit
takes under a second. Attack and defence are centred at zero with a division
intercept carrying the overall scoring rate.

Four things beyond textbook Dixon-Coles:

**Time decay.** Each match is weighted `exp(−ln2 · age_days / half_life)`.

**Ridge shrinkage.** A Gaussian prior pulls ratings toward league average,
strength set by `--prior-sd`.

**Joint two-division fit.** The Premier League and Championship are fitted
together. Team strength belongs to the team, not the division, so clubs that
recently moved between them bridge the two and promoted sides get real priors.

**Mixed likelihood on xG.** With `--target xg` or `blend`, matches that have xG
use a Gamma likelihood on the expected-goals value; matches without fall back
to Poisson on actual goals. Both feed the same linear predictor, and the Gamma
shape parameter is estimated, so the relative weight of an xG match versus a
goals-only match comes out of the data. Since τ describes discrete scorelines,
ρ is re-estimated from actual goals afterward.

### The season simulation

A league season is a complete double round robin, so remaining fixtures are
every ordered pair of clubs minus those already played. No fixture feed needed.
Each simulation samples a scoreline for every remaining fixture, adds them to
points already banked, and ranks with real tiebreakers. Twenty thousand runs
gives the probability of each outcome.

---

## 7. What we built, in order

**The Dixon-Coles engine and the season simulator.** Analytic gradients from
the start, verified against numerical differentiation to about 1e-6, which is
why fits take a fraction of a second and a full backtest takes seconds rather
than minutes.

**Walk-forward backtesting.** Refit weekly on only prior data, predict, score.
Ranked Probability Score as the primary metric, with the bookmaker's closing
line alongside as the benchmark.

**Expected goals.** Added a Gamma likelihood so ratings could be fitted on xG,
which is far less noisy than goals. FBref turned out to be unusable, so the
pipeline runs on Understat. Verified on synthetic data that fitting on xG
roughly halves the error in recovered team strength.

**Market blending.** Logarithmic and linear pooling with a fitted weight.

**Uncertainty in the projections.** A bootstrap for rating uncertainty,
mean-reverting drift for ratings changing over a season, and finally a
calibrated post-hoc widening.

### Things that changed along the way

| Change | Why |
|---|---|
| FBref → Understat for xG | FBref returned schedules with the xG columns missing. It drives a real Chrome browser via seleniumbase, which also tripped macOS App Management protection. Understat is plain JSON over HTTP. |
| SSL certificate handling | macOS Python ships an empty certificate store, so every HTTPS request failed. The downloader now builds its context from certifi. |
| Byte-order mark stripping | The fixtures file is UTF-8 with a BOM, read as latin-1, which glued three junk bytes onto the first column name. |
| Fixtures filtered against results | The upcoming-fixtures feed is a rolling window that still carries the round just played, so the card was showing finished matches. |
| `--odds closing/opening` as a flag | Originally required hand-editing a list, and an unsaved edit produced a silently identical result. |
| Tune grid made configurable | The first grid's optimum sat on its boundary in both directions. |
| Drift reparametrised | Calibrating a step size and a reversion rate separately is unidentifiable from end-of-season coverage. The long-run spread is the quantity coverage can actually pin down. |
| Drift estimator scrapped entirely | See section 8. |

---

## 8. What's proven and what isn't

### xG helps: proven

| Season | goals | xg | blend | market |
|---|---|---|---|---|
| 2023-24 | 0.1957 | 0.1935 | 0.1926 | 0.1802 |
| 2024-25 | 0.2081 | 0.2013 | 0.2023 | 0.1962 |
| 2025-26 | 0.2120 | 0.2090 | 0.2088 | 0.2048 |

xG beats goals in all three seasons, on every metric. Blend versus xG flips
between seasons and the margins are around 0.001, so treat those two as tied.
Keep `--target blend`.

### Tuning doesn't transfer: proven

The grid search gained 0.0046 RPS on the season it searched and 0.0001 on a
held-out one. The surface is flat. Keep the defaults and stop thinking about
them.

### Market blending is a dead end: proven

The fitted weight came out 0.000 against both closing and opening odds. Across
760 matches the model adds nothing to the bookmaker's line. Opening odds score
0.2057 and closing 0.2048, so the market improves by only 0.0009 across the
week before kickoff. Its advantage isn't late team news; it's better from the
moment the line opens.

This is why the match predictions are for interest and the season projections
are the actual product. Nobody publishes a free calibrated distribution over
final league positions, and the market's short-horizon edge largely evaporates
over a nine-month question.

### The projections were overconfident: proven

Standing at matchweek 4 across seven seasons, 140 club-seasons, only 68.6% of
clubs finished inside a band meant to hold 80%. Roughly one in three landed
outside.

### The ratings are unbiased: proven

Mean PIT sat between 0.495 and 0.506 at every setting tested. Nobody was rated
systematically too high or too low. Only the spread was wrong, which is the
good version of this problem.

### Why the spread is wrong: unresolved

Three hypotheses tested, two eliminated:

- **Rating drift.** Reaching 80% coverage needed a long-run spread of 0.470,
  three times the entire spread of team strengths in the league. Physically
  absurd, and at that setting clubs on one point kept 1%+ title chances.
  Rejected.
- **Over-shrinkage.** Halving it moved coverage from 68.6% to 67.9%, the wrong
  direction. Rejected.
- **The bootstrap understates uncertainty.** Untested, and now the prime
  suspect. It moved the mean points span from 18.1 to 18.6, which is almost
  nothing. Resampling matches disturbs the time-decay weighting the model
  depends on, and a bootstrap can only see sampling variability given the model
  is correct, never misspecification.

The unresolved part is why `--widen 1.5` exists. It's a stated correction with
no claimed mechanism. Coverage runs 75.0, 79.3, 83.6, 86.4, 90.7 across
widening factors 1.4 to 1.8, so 1.5 is a genuine interior optimum.

The cost: widening gives bad clubs a small path upward. At 1.5 the bottom five
collectively hold about 0.6% of title probability. Coventry on zero points
shows 0.2%, which is the least convincing number in the output.

### The unfinished thread

Fit the model on two independent halves of the data and compare how far apart
those ratings land against the spread across bootstrap resamples. If the
bootstrap spread is much smaller, the diagnosis is confirmed and the fix is to
resample in a way that respects time weighting. That would replace the 1.5
multiplier with an actual mechanism.

---

## 9. Weekly routine

```bash
cd ~/Projects/PL_2026-27_Predictions
source .venv/bin/activate                                      # every new terminal

python run.py table                                            # results update automatically
python run.py card --out card.csv                              # this week's fixtures
python run.py simulate --sims 20000 --bootstrap 40 --widen 1.5 --out projections.csv
```

Refresh xG every few weeks with `python run.py fetch-xg --refresh`, since the
current season's file needs re-pulling to pick up new matches.

Re-run `drift --mode widen` occasionally, maybe every couple of months. As the
season progresses and ratings rest on more current-season data, the required
widening should shrink. If it doesn't, that's informative in itself.

---

## 10. Reading the numbers honestly

**Probabilities are not predictions.** A 52% title chance means the model
expects to be wrong just under half the time.

**The intervals are corrected, not derived.** With `--widen 1.5` they're
calibrated against seven seasons of history, but through a correction whose
cause is unknown.

**Point estimates are more reliable than tails.** The ordering and the
projected points are solid and unbiased. A 0.2% title chance for a club on zero
points is an artifact of the correction, not a real assessment.

**Promoted clubs are the weakest part.** Hull, Ipswich and Coventry have no xG
history, because Understat doesn't cover the Championship. Their ratings come
from goals-only second-tier data. If the model is wrong anywhere right now,
it's most likely there.

**The bookmaker beats you at match prediction.** Measured, repeatedly. The
model is for understanding the season, not for betting into a market that has
already proven sharper.