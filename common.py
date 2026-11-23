"""Shared helpers: loading processed data, window counts, design matrices, statistics."""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config as C


# ----------------------------------------------------------------------------
# Loading processed insertions
# ----------------------------------------------------------------------------
def list_insertions():
    return sorted(p for p in C.PROCESSED_DIR.iterdir() if (p / 'meta.json').exists())


def load_insertion(path):
    meta = json.loads((path / 'meta.json').read_text())
    trials = pd.read_parquet(path / 'trials.parquet')
    units = pd.read_parquet(path / 'units.parquet')
    b = np.load(path / 'binned.npz')
    binned = {k: b[k] for k in b.files}
    return meta, trials, units, binned


def bin_centers():
    edges = np.round(np.arange(C.EDGES[0], C.EDGES[1] + C.BIN / 2, C.BIN), 6)
    return edges, (edges[:-1] + edges[1:]) / 2


def window_counts(binned, window):
    """Spike counts (units x trials) summed over an analysis window from config.WINDOWS."""
    event, t0, t1 = C.WINDOWS[window] if isinstance(window, str) else window
    key = {'stimOn': 'so_counts', 'firstMove': 'fm_counts'}[event]
    edges, _ = bin_centers()
    i0 = int(round((t0 - edges[0]) / C.BIN))
    i1 = int(round((t1 - edges[0]) / C.BIN))
    return binned[key][:, :, i0:i1].sum(-1).astype(float)


# ----------------------------------------------------------------------------
# Design matrix
# ----------------------------------------------------------------------------
def _z(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else np.zeros_like(x)


def design_matrix(trials, window, include_me=True):
    """
    Regressors for the single-unit GLM. Returns (X, column_names, term_index_dict).

    choice     : reported choice (+1/-1, IBL convention)
    stimulus   : stimulus side (sign of signed contrast; 0 on 0% trials) and |contrast|
    prior      : block prior, probabilityLeft - 0.5
    speed      : peak wheel speed, linear AND log (tested jointly, 2 df)
    rt, dur    : log reaction time, movement duration
    me         : whisker & body video motion energy in the same window (if available)
    drift      : trial number, linear and quadratic (slow non-stationarity)
    """
    t = trials
    cols = {
        'intercept': np.ones(len(t)),
        'choice': t['choice'].to_numpy(float),
        'stim_side': np.sign(t['signed_contrast'].to_numpy()),
        'stim_abs': _z(np.abs(t['signed_contrast'].to_numpy())),
        'prior': t['probabilityLeft'].to_numpy(float) - 0.5,
        'speed_lin': _z(t['peak_speed']),
        'speed_log': _z(np.log(t['peak_speed'].to_numpy() + 1e-3)),
        'rt': _z(np.log(np.clip(t['rt'].to_numpy(), 1e-3, None))),
        'dur': _z(np.log(np.clip(t['move_dur'].to_numpy(), 1e-3, None))),
        'drift_lin': _z(t['trial_number']),
    }
    cols['drift_quad'] = _z(cols['drift_lin'] ** 2)
    terms = {'choice': ['choice'], 'stimulus': ['stim_side', 'stim_abs'], 'prior': ['prior'],
             'speed': ['speed_lin', 'speed_log'], 'rt': ['rt'], 'dur': ['dur']}
    if include_me:
        me_cols = []
        for cam in ('whisker', 'body'):
            c = f'me_{cam}_{window}'
            if c in t.columns and t[c].notna().mean() > 0.9:
                v = t[c].to_numpy(float)
                v = np.where(np.isnan(v), np.nanmean(v), v)
                cols[f'me_{cam}'] = _z(v)
                me_cols.append(f'me_{cam}')
        if me_cols:
            terms['me'] = me_cols
    names = list(cols)
    X = np.column_stack([cols[n] for n in names])
    idx = {k: [names.index(c) for c in v] for k, v in terms.items()}
    return X, names, idx


def valid_trials(trials):
    ok = np.isfinite(trials['peak_speed']) & (trials['peak_speed'] > 0) & np.isfinite(trials['move_dur'])
    return ok.to_numpy()


def vif(X, names):
    """Variance inflation factors (excluding intercept)."""
    out = {}
    for j, n in enumerate(names):
        if n == 'intercept':
            continue
        others = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        r = X[:, j] - others @ beta
        r2 = 1 - r.var() / X[:, j].var() if X[:, j].var() > 0 else 1
        out[n] = 1 / max(1 - r2, 1e-9)
    return out


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def fdr_bh(p):
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(n)
    qq[order] = np.clip(ranked, 0, 1)
    q[ok] = qq
    return q


def binom_ci(k, n, level=0.95):
    if n == 0:
        return np.nan, np.nan
    r = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=level)
    return r.low, r.high


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.transparent': False})
    return plt


COL_CHOICE = '#2a6f97'
COL_SPEED = '#c97b2a'
COL_PRIOR = '#6a994e'
COL_STIM = '#8e5ea2'
COL_GREY = '0.55'
