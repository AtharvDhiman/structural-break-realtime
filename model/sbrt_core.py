"""
Structural Break: Real-Time — core engine (shared by train and infer).

Self-contained: AR(2)+GARCH(1,1) residual whitening, four calibrated residual
statistics (each = running mean of a feature of the standardized residual z,
turned into an empirical quantile against length-matched history windows), and
an incremental per-point scorer whose output is bit-identical to a batch pass.

Depends only on numpy / statsmodels / arch. No global RNG is used, so results are
deterministic and independent of series processing order.
"""
import math
import warnings
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")

GRID = np.array([5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000])
MIN_WINDOWS = 20
STAT_NAMES = ["mean", "var", "tail", "acf1"]
# feature-vector order the LightGBM model expects
FEAT_ORDER = (["s_" + n for n in STAT_NAMES] + ["m_" + n for n in STAT_NAMES] + ["t"])
AR_ORDER = 2

# ----------------------------------------------------------------- fitting
def fit_params(hist, p=AR_ORDER):
    """AR(p)+GARCH(1,1) on the historical segment. Pure function of `hist`."""
    hist = np.asarray(hist, dtype=np.float64)
    H = len(hist)
    p = min(p, max(1, H // 100))
    ar = AutoReg(hist, lags=p, trend="c", old_names=False).fit()
    c = float(ar.params[0])
    phi = np.asarray(ar.params[1:], dtype=np.float64)
    e_hist = np.asarray(ar.resid, dtype=np.float64)
    omega, alpha, beta = _fit_garch(e_hist)
    h_hist = _garch_var_path(e_hist, omega, alpha, beta)
    z_hist = e_hist / np.sqrt(h_hist)
    return {
        "p": p, "c": c, "phi": phi,
        "omega": omega, "alpha": alpha, "beta": beta,
        "seed_x": hist[-p:].copy(), "seed_e2": float(e_hist[-1] ** 2),
        "seed_h": float(h_hist[-1]), "z_hist": z_hist,
    }


def _fit_garch(e):
    from arch import arch_model
    try:
        res = arch_model(e, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(
            disp="off", show_warning=False)
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        if not np.isfinite([omega, alpha, beta]).all() or alpha + beta >= 0.999 or omega <= 0:
            raise ValueError
        return omega, alpha, beta
    except Exception:
        return max(float(np.var(e)), 1e-8), 0.0, 0.0   # floor: avoid zero-variance path


def _garch_var_path(e, omega, alpha, beta):
    n = len(e)
    h = np.empty(n)
    h[0] = max(np.var(e), 1e-8) if (alpha + beta) == 0 else omega / max(1e-8, 1 - alpha - beta)
    for t in range(1, n):
        h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
    return h


# ----------------------------------------------------------------- features
def feature_hist(name, z):
    if name == "mean": return z
    if name == "var":  return z ** 2
    if name == "tail": return (np.abs(z) > 2).astype(np.float64)
    if name == "acf1": return z[1:] * z[:-1]


def feature_point(name, z, z_prev):
    if name == "mean": return z
    if name == "var":  return z * z
    if name == "tail": return 1.0 if abs(z) > 2 else 0.0
    if name == "acf1": return z * z_prev


# ----------------------------------------------------------------- null / calibration
def build_null(f, grid=GRID, min_windows=MIN_WINDOWS):
    f = np.asarray(f, dtype=np.float64)
    H = len(f)
    cf = np.concatenate([[0.0], np.cumsum(f)])
    null = {}
    for L in grid:
        L = int(L)
        if L > H or (H - L + 1) < min_windows:
            continue
        null[L] = np.sort((cf[L:] - cf[:-L]) / L)
    return np.array(sorted(null.keys())), null


def _q_at(null, L, S):
    arr = null[int(L)]
    return np.searchsorted(arr, S, side="right") / len(arr)


def calibrate_point(S, t, valid_Ls, null):
    """Two-sided empirical quantile of running-mean S at step t, interpolated
    across the length grid. Identical arithmetic to the batch version."""
    Ls = valid_Ls
    if t <= Ls[0]:
        q = _q_at(null, Ls[0], S)
    elif t >= Ls[-1]:
        q = _q_at(null, Ls[-1], S)
    else:
        i = int(np.searchsorted(Ls, t))
        Llo, Lhi = Ls[i - 1], Ls[i]
        w = (t - Llo) / (Lhi - Llo)
        q = (1 - w) * _q_at(null, Llo, S) + w * _q_at(null, Lhi, S)
    return 2.0 * abs(q - 0.5)


# ----------------------------------------------------------------- incremental scorer (infer)
class SeriesScorer:
    """One per series. prepare(historical) once, then update(x) per online point."""
    def __init__(self, booster):
        self.booster = booster

    def prepare(self, historical):
        pr = fit_params(historical)
        self.pr = pr
        z_hist = pr["z_hist"]
        self.nulls = {n: build_null(feature_hist(n, z_hist)) for n in STAT_NAMES}
        self.t = 0
        self.sums = {n: 0.0 for n in STAT_NAMES}
        self.maxes = {n: 0.0 for n in STAT_NAMES}
        self.xbuf = list(pr["seed_x"])
        self.e_prev2 = pr["seed_e2"]
        self.h_prev = pr["seed_h"]
        self.z_prev = float(z_hist[-1]) if len(z_hist) else 0.0

    def update(self, x):
        pr = self.pr; p = pr["p"]
        pred = pr["c"] + float(np.dot(pr["phi"], self.xbuf[-p:][::-1]))
        e = x - pred
        h = pr["omega"] + pr["alpha"] * self.e_prev2 + pr["beta"] * self.h_prev
        z = e / math.sqrt(max(h, 1e-12))
        self.xbuf.append(x); self.e_prev2 = e * e; self.h_prev = h
        self.t += 1
        s_vals, m_vals = [], []
        for n in STAT_NAMES:
            self.sums[n] += feature_point(n, z, self.z_prev)
            S = self.sums[n] / self.t
            valid, null = self.nulls[n]
            sc = calibrate_point(S, self.t, valid, null) if len(valid) else 0.5
            if sc > self.maxes[n]:
                self.maxes[n] = sc
            s_vals.append(sc); m_vals.append(self.maxes[n])
        self.z_prev = z
        row = np.array(s_vals + m_vals + [float(self.t - 1)], dtype=np.float64).reshape(1, -1)
        out = float(self.booster.predict(row)[0])
        return out if np.isfinite(out) else 0.5   # never emit NaN/inf


# ----------------------------------------------------------------- batch features (train)
def series_features(historical, online):
    """Batch feature matrix for one series (training). Columns follow FEAT_ORDER."""
    pr = fit_params(historical)
    z = _online_z(pr, online)
    n = len(online)
    s_cols, m_cols = [], []
    for name in STAT_NAMES:
        fh = feature_hist(name, pr["z_hist"])
        valid, null = build_null(fh)
        prev = pr["z_hist"][-1] if len(pr["z_hist"]) else 0.0
        zprev = np.concatenate([[prev], z[:-1]]) if n else z
        fo = feature_point_vec(name, z, zprev)
        if len(valid) == 0 or n == 0:
            sc = np.full(n, 0.5)
        else:
            sc = _calibrate_vec(fo, valid, null)
        s_cols.append(sc); m_cols.append(np.maximum.accumulate(sc) if n else sc)
    t_col = np.arange(n, dtype=np.float64)
    return np.column_stack(s_cols + m_cols + [t_col])


def feature_point_vec(name, z, z_prev):
    if name == "mean": return z
    if name == "var":  return z ** 2
    if name == "tail": return (np.abs(z) > 2).astype(np.float64)
    if name == "acf1": return z * z_prev


def _calibrate_vec(f_online, valid_Ls, null):
    t = np.arange(1, len(f_online) + 1)
    S = np.cumsum(f_online) / t
    Q = np.empty((len(valid_Ls), len(S)))
    for k, L in enumerate(valid_Ls):
        arr = null[int(L)]
        Q[k] = np.searchsorted(arr, S, side="right") / len(arr)
    idx = np.searchsorted(valid_Ls, t)
    q = np.empty(len(S))
    below = np.where(idx == 0)[0]; above = np.where(idx == len(valid_Ls))[0]
    mid = np.where((idx > 0) & (idx < len(valid_Ls)))[0]
    q[below] = Q[0, below]; q[above] = Q[-1, above]
    if mid.size:
        hi = idx[mid]; lo = hi - 1
        Llo = valid_Ls[lo]; Lhi = valid_Ls[hi]
        w = (t[mid] - Llo) / (Lhi - Llo)
        q[mid] = (1 - w) * Q[lo, mid] + w * Q[hi, mid]
    return 2.0 * np.abs(q - 0.5)


def _online_z(params, online):
    online = np.asarray(online, dtype=np.float64)
    n = len(online)
    p = params["p"]; c = params["c"]; phi = params["phi"]
    omega = params["omega"]; alpha = params["alpha"]; beta = params["beta"]
    xbuf = list(params["seed_x"]); e_prev2 = params["seed_e2"]; h_prev = params["seed_h"]
    z = np.empty(n)
    for t in range(n):
        x = online[t]
        pred = c + float(np.dot(phi, xbuf[-p:][::-1]))
        e = x - pred
        h = omega + alpha * e_prev2 + beta * h_prev
        z[t] = e / math.sqrt(max(h, 1e-12))
        xbuf.append(x); e_prev2 = e * e; h_prev = h
    return z
