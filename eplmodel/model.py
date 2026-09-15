"""Dixon-Coles goal model, with optional fitting on expected goals.

    lambda_home = exp(mu_div + home_adv + attack_home - defence_away)
    lambda_away = exp(mu_div           + attack_away - defence_home)

Match probabilities always come from a Dixon-Coles score matrix built on those
rates: Poisson marginals plus the tau correction for 0-0, 1-0, 0-1 and 1-1.
What changes with `target` is how the rates themselves are estimated.

  target="goals"  Poisson likelihood on actual goals. The classic.
  target="xg"     Gamma likelihood on expected goals. xG is a much less noisy
                  measurement of the same underlying rate, so ratings converge
                  in roughly half the matches.
  target="blend"  Gamma likelihood on w*xG + (1-w)*goals. Usually the best of
                  the three: xG discards finishing skill, and goals carry some
                  real signal about it.

Matches with no xG available fall back to the Poisson term automatically, so
you can fit across seasons and divisions with patchy xG coverage in one go.
The two likelihoods share the same linear predictor, and the Gamma shape is
estimated, so the relative weight of an xG match against a goals-only match
comes out of the data rather than being asserted.

Attack and defence are centred at zero; a division intercept carries the
overall scoring rate. Higher defence means a better defence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import digamma, gammaln

MAX_GOALS = 12
XG_FLOOR = 0.05  # Gamma has no mass at zero; a goalless, shotless half is rare
RHO_BOUND = 0.20


@dataclass
class DixonColes:
    half_life_days: float = 240.0
    prior_sd: float = 0.40
    max_goals: int = MAX_GOALS
    target: str = "goals"
    blend_weight: float = 0.70

    teams: list[str] = field(default_factory=list)
    divisions: list[str] = field(default_factory=list)
    attack: np.ndarray | None = None
    defence: np.ndarray | None = None
    home_adv: float = 0.0
    rho: float = 0.0
    phi: float = float("nan")
    div_intercept: np.ndarray | None = None
    n_matches: int = 0
    n_xg_matches: int = 0
    converged: bool = False
    loglik: float = float("nan")

    # ---------------------------------------------------------------- fitting

    def _observations(self, matches: pd.DataFrame):
        """Returns (obs_home, obs_away, use_gamma_mask)."""
        x = matches["hg"].to_numpy(dtype=np.float64)
        y = matches["ag"].to_numpy(dtype=np.float64)
        if self.target == "goals":
            return x, y, np.zeros(len(matches), dtype=bool)
        if self.target not in {"xg", "blend"}:
            raise ValueError(f"target must be 'goals', 'xg' or 'blend', got {self.target!r}")
        if "xg_h" not in matches.columns:
            raise ValueError("target requires xg_h and xg_a columns; run fetch-xg first")

        xh = matches["xg_h"].to_numpy(dtype=np.float64)
        xa = matches["xg_a"].to_numpy(dtype=np.float64)
        use = np.isfinite(xh) & np.isfinite(xa)
        if not use.any():
            raise ValueError("target is xg/blend but no match has xG; run fetch-xg first")

        oh, oa = x.copy(), y.copy()
        if self.target == "xg":
            oh[use], oa[use] = xh[use], xa[use]
        else:
            w = self.blend_weight
            oh[use] = w * xh[use] + (1 - w) * x[use]
            oa[use] = w * xa[use] + (1 - w) * y[use]
        oh[use] = np.clip(oh[use], XG_FLOOR, None)
        oa[use] = np.clip(oa[use], XG_FLOOR, None)
        return oh, oa, use

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None, verbose: bool = False) -> "DixonColes":
        matches = matches.dropna(subset=["home", "away", "hg", "ag", "date"])
        if as_of is not None:
            matches = matches[matches["date"] < as_of]
        if len(matches) < 50:
            raise ValueError(f"need at least 50 matches to fit, got {len(matches)}")

        self.teams = sorted(set(matches["home"]) | set(matches["away"]))
        self.divisions = sorted(matches["div"].astype(str).unique())
        t_index = {t: i for i, t in enumerate(self.teams)}
        d_index = {d: i for i, d in enumerate(self.divisions)}
        n_t, n_d = len(self.teams), len(self.divisions)

        hi = matches["home"].map(t_index).to_numpy(dtype=np.int64)
        ai = matches["away"].map(t_index).to_numpy(dtype=np.int64)
        di = matches["div"].astype(str).map(d_index).to_numpy(dtype=np.int64)
        gx = matches["hg"].to_numpy(dtype=np.int64)
        gy = matches["ag"].to_numpy(dtype=np.int64)

        obs_h, obs_a, gam = self._observations(matches)
        poi = ~gam
        self.n_matches, self.n_xg_matches = len(matches), int(gam.sum())

        ref = as_of if as_of is not None else matches["date"].max()
        age_days = (ref - matches["date"]).dt.total_seconds().to_numpy() / 86400.0
        w = np.exp(-math.log(2.0) / self.half_life_days * np.clip(age_days, 0.0, None))

        # Poisson pieces
        px, py = obs_h[poi], obs_a[poi]
        wp = w[poi]
        poisson_const = -(gammaln(px + 1.0) + gammaln(py + 1.0)) * wp
        lx, ly = gx[poi], gy[poi]
        m00 = (lx == 0) & (ly == 0)
        m01 = (lx == 0) & (ly == 1)
        m10 = (lx == 1) & (ly == 0)
        m11 = (lx == 1) & (ly == 1)

        # Gamma pieces
        gxh, gxa = obs_h[gam], obs_a[gam]
        wg = w[gam]
        log_gxh, log_gxa = np.log(gxh), np.log(gxa)

        prec = 1.0 / (self.prior_sd**2)
        fit_phi = bool(gam.any())
        n_par = 2 * n_t + 2 + n_d + (1 if fit_phi else 0)

        def unpack(p):
            att = p[:n_t]
            dfn = p[n_t : 2 * n_t]
            mu = p[2 * n_t + 2 : 2 * n_t + 2 + n_d]
            phi = math.exp(p[-1]) if fit_phi else float("nan")
            return att - att.mean(), dfn - dfn.mean(), p[2 * n_t], p[2 * n_t + 1], mu, phi

        def objective(p):
            att, dfn, home, rho, mu, phi = unpack(p)
            e1_all = mu[di] + home + att[hi] - dfn[ai]
            e2_all = mu[di] + att[ai] - dfn[hi]

            gatt = np.zeros(n_t)
            gdfn = np.zeros(n_t)
            gmu = np.zeros(n_d)
            ghome = 0.0
            grho = 0.0
            gphi_log = 0.0
            ll = 0.0

            # ---- Poisson block (with the Dixon-Coles tau correction)
            if poi.any():
                e1, e2 = e1_all[poi], e2_all[poi]
                lam, mv = np.exp(e1), np.exp(e2)
                tau = np.ones_like(lam)
                tau[m00] = 1.0 - lam[m00] * mv[m00] * rho
                tau[m01] = 1.0 + lam[m01] * rho
                tau[m10] = 1.0 + mv[m10] * rho
                tau[m11] = 1.0 - rho
                tau = np.clip(tau, 1e-9, None)

                ll += np.sum(wp * (px * e1 - lam + py * e2 - mv + np.log(tau))) + poisson_const.sum()

                dtl = np.zeros_like(lam)
                dtm = np.zeros_like(lam)
                drho = np.zeros_like(lam)
                lm = lam * mv
                dtl[m00] = -lm[m00] * rho
                dtm[m00] = -lm[m00] * rho
                drho[m00] = -lm[m00] / tau[m00]
                dtl[m01] = lam[m01] * rho
                drho[m01] = lam[m01] / tau[m01]
                dtm[m10] = mv[m10] * rho
                drho[m10] = mv[m10] / tau[m10]
                drho[m11] = -1.0 / tau[m11]

                g1 = wp * (px - lam + dtl / tau)
                g2 = wp * (py - mv + dtm / tau)
                np.add.at(gatt, hi[poi], g1)
                np.add.at(gatt, ai[poi], g2)
                np.add.at(gdfn, ai[poi], -g1)
                np.add.at(gdfn, hi[poi], -g2)
                np.add.at(gmu, di[poi], g1 + g2)
                ghome += g1.sum()
                grho += np.sum(wp * drho)

            # ---- Gamma block (continuous xG or blended target)
            if fit_phi:
                e1, e2 = e1_all[gam], e2_all[gam]
                lam, mv = np.exp(e1), np.exp(e2)
                term = (
                    (phi - 1.0) * (log_gxh + log_gxa)
                    - phi * (gxh / lam + gxa / mv)
                    - phi * (e1 + e2)
                    + 2.0 * (phi * math.log(phi) - gammaln(phi))
                )
                ll += np.sum(wg * term)

                g1 = wg * phi * (gxh / lam - 1.0)
                g2 = wg * phi * (gxa / mv - 1.0)
                np.add.at(gatt, hi[gam], g1)
                np.add.at(gatt, ai[gam], g2)
                np.add.at(gdfn, ai[gam], -g1)
                np.add.at(gdfn, hi[gam], -g2)
                np.add.at(gmu, di[gam], g1 + g2)
                ghome += g1.sum()

                dphi = (
                    (log_gxh + log_gxa)
                    - (gxh / lam + gxa / mv)
                    - (e1 + e2)
                    + 2.0 * (math.log(phi) + 1.0 - digamma(phi))
                )
                gphi_log = phi * np.sum(wg * dphi)

            gatt -= gatt.mean()   # chain rule through the centring
            gdfn -= gdfn.mean()

            raw_att, raw_dfn = p[:n_t], p[n_t : 2 * n_t]
            pen = 0.5 * prec * (np.sum(raw_att**2) + np.sum(raw_dfn**2))

            parts = [gatt - prec * raw_att, gdfn - prec * raw_dfn, [ghome], [grho], gmu]
            if fit_phi:
                parts.append([gphi_log])
            return -(ll - pen), -np.concatenate(parts)

        p0 = np.concatenate([np.zeros(2 * n_t), [0.25], [-0.05], np.full(n_d, 0.1)])
        bounds = [(-3.0, 3.0)] * (2 * n_t) + [(-1.0, 1.0), (-RHO_BOUND, RHO_BOUND)] + [(-2.0, 2.0)] * n_d
        if fit_phi:
            p0 = np.append(p0, math.log(3.0))
            bounds.append((math.log(0.3), math.log(80.0)))
        assert len(p0) == n_par

        res = minimize(objective, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 1000, "ftol": 1e-10})

        att, dfn, home, rho, mu, phi = unpack(res.x)
        self.attack, self.defence = att, dfn
        self.home_adv, self.rho, self.div_intercept, self.phi = float(home), float(rho), mu, phi
        self.converged = bool(res.success)
        self.loglik = float(-res.fun)

        # The tau correction describes the discrete scoreline, so when the rates
        # were fitted on xG, rho has to be re-estimated from actual goals.
        if fit_phi:
            self.rho = self._profile_rho(hi, ai, di, gx, gy, w)

        if verbose:
            cov = f"{self.n_xg_matches}/{self.n_matches} with xG" if self.n_xg_matches else "goals only"
            print(f"fit on {self.n_matches} matches | teams={n_t} | target={self.target} | {cov} | converged={self.converged}")
            line = f"  home advantage {self.home_adv:+.3f}  rho {self.rho:+.3f}"
            if fit_phi:
                line += f"  gamma shape {self.phi:.2f}"
            print(line)
            print(f"  half-life {self.half_life_days:.0f}d  prior sd {self.prior_sd:.2f}")
        return self

    def _profile_rho(self, hi, ai, di, gx, gy, w) -> float:
        lam = np.exp(self.div_intercept[di] + self.home_adv + self.attack[hi] - self.defence[ai])
        mv = np.exp(self.div_intercept[di] + self.attack[ai] - self.defence[hi])
        m00 = (gx == 0) & (gy == 0)
        m01 = (gx == 0) & (gy == 1)
        m10 = (gx == 1) & (gy == 0)
        m11 = (gx == 1) & (gy == 1)

        def neg(rho):
            tau = np.ones_like(lam)
            tau[m00] = 1.0 - lam[m00] * mv[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mv[m10] * rho
            tau[m11] = 1.0 - rho
            return -np.sum(w * np.log(np.clip(tau, 1e-9, None)))

        r = minimize_scalar(neg, bounds=(-RHO_BOUND, RHO_BOUND), method="bounded")
        return float(r.x)

    # ------------------------------------------------------------ prediction

    def _rates(self, home: str, away: str, div: str = "E0") -> tuple[float, float]:
        if self.attack is None:
            raise RuntimeError("model is not fitted")
        idx = {t: i for i, t in enumerate(self.teams)}
        for t in (home, away):
            if t not in idx:
                raise KeyError(f"unknown team {t!r}; not present in the training data")
        d = self.divisions.index(div) if div in self.divisions else 0
        h, a = idx[home], idx[away]
        lam = math.exp(self.div_intercept[d] + self.home_adv + self.attack[h] - self.defence[a])
        mv = math.exp(self.div_intercept[d] + self.attack[a] - self.defence[h])
        return lam, mv

    def score_matrix(self, home: str, away: str, div: str = "E0") -> np.ndarray:
        """P(home goals = i, away goals = j), normalised."""
        lam, mv = self._rates(home, away, div)
        k = np.arange(self.max_goals + 1)
        ph = np.exp(k * math.log(lam) - lam - gammaln(k + 1.0))
        pa = np.exp(k * math.log(mv) - mv - gammaln(k + 1.0))
        m = np.outer(ph, pa)
        m[0, 0] *= 1.0 - lam * mv * self.rho
        m[0, 1] *= 1.0 + lam * self.rho
        m[1, 0] *= 1.0 + mv * self.rho
        m[1, 1] *= 1.0 - self.rho
        m = np.clip(m, 0.0, None)
        return m / m.sum()

    def predict(self, home: str, away: str, div: str = "E0") -> dict:
        m = self.score_matrix(home, away, div)
        lam, mv = self._rates(home, away, div)
        k = np.arange(self.max_goals + 1)
        totals = k[:, None] + k[None, :]
        i, j = np.unravel_index(np.argsort(m, axis=None)[::-1][:3], m.shape)
        return {
            "home": home,
            "away": away,
            "p_home": float(np.tril(m, -1).sum()),
            "p_draw": float(np.trace(m)),
            "p_away": float(np.triu(m, 1).sum()),
            "xg_home": lam,
            "xg_away": mv,
            "p_over_2_5": float(m[totals > 2.5].sum()),
            "p_btts": float(m[1:, 1:].sum()),
            "top_scores": [(int(a), int(b), float(m[a, b])) for a, b in zip(i, j)],
        }

    def ratings(self) -> pd.DataFrame:
        df = pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence})
        df["overall"] = df["attack"] + df["defence"]
        return df.sort_values("overall", ascending=False).reset_index(drop=True)


def outcome_probabilities(model: DixonColes, fixtures, div: str = "E0") -> pd.DataFrame:
    rows = [model.predict(h, a, div) for h, a in fixtures]
    df = pd.DataFrame(rows)
    return df[["home", "away", "p_home", "p_draw", "p_away", "xg_home", "xg_away", "p_over_2_5", "p_btts"]]