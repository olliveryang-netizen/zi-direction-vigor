"""
Step 14 - Publication figures for Journal of Neurophysiology (APS).

Draws every figure of the paper from the results saved by steps 02-13. Nothing is refitted.
Behaviour and example units are read from processed/ (ZI). Run it from the project folder:

    python 14_make_figures.py            # all figures
    python 14_make_figures.py --only 4   # one main figure (1-6) or S1-S5

Format (APS "Preparing figures"): full page width 7.0 in, at most 9 in deep. There is one
sans-serif font (Helvetica Bold; Arial Bold if Helvetica cannot be found). Panel letters are bold A, B, C.
Each figure is saved as:
    figures_paper/FigN.pdf   vector, fonts embedded (TrueType), for submission
    figures_paper/FigN.tif   600 dpi, LZW (for systems that ask for TIFF)
    figures_paper/FigN.png   300 dpi, for viewing
plus figures_paper/figure_legends.md: the numbers (n, means, tests) for writing the legends.

Each figure is checked for overlapping text. Any overlap is printed as a WARNING.

Colour code (the same in every figure)
    direction / right choice  blue     vigour (speed)    amber     prior     green
    left choice               red      body pose         brown     imposter  grey
    regions: ZI blue, MOs teal, VPM purple
"""
import argparse
import io
import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                   # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402
from matplotlib.ticker import MaxNLocator              # noqa: E402

import config as C                                    # noqa: E402
import common as K                                    # noqa: E402

warnings.filterwarnings('ignore')
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

PROJ = C.PROJECT_DIR
OUT = PROJ / 'figures_paper'
CMP = PROJ / 'results_comparison'
WIDTH = 7.0                                           # inches, APS full width (6-7.15 in)
ORDER = ['ZI', 'MOs', 'VPM', 'SCm', 'MRN', 'APN', 'RSPd', 'VISp']

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
DIR = '#1F4E8C'          # direction information, right choice, ZI
LEFT = '#B23A2E'         # left choice
VIG = '#E0951F'          # vigour (speed)
VIG_SH = ['#F3CF8B', '#E0951F', '#9C5A0C']      # slow / medium / fast
PRIOR = '#4E8A3E'
STIM = '#7C5BA6'
POSE = '#7A5A3A'
MIX = '#6E6E9E'
NULL_F = '#D9D9D9'       # imposter fill
NULL_L = '#8A8A8A'       # imposter line
DOT = '#A9A9A9'          # individual sessions
INK = '#1A1A1A'
REG_COL = {'ZI': DIR, 'MOs': '#2B8A78', 'VPM': '#8A63A8'}
REG_EXTRA = ['#B5651D', '#5C7A99', '#9E9E3A', '#A0526D', '#4D4D4D']

FS = dict(tick=8.5, label=9.5, legend=8.0, annot=8.5, letter=14.0)

MISSING = []
LEGENDS = {}
WARN = []


# ----------------------------------------------------------------------------
# Font and style
# ----------------------------------------------------------------------------
def _try_font(path):
    font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    fig = plt.figure(figsize=(1, 1))
    fig.text(0.1, 0.5, 'Test \u22120.5 R\u00b2', family=name, weight='bold')
    fig.savefig(io.BytesIO(), format='pdf')
    plt.close(fig)
    return name


def _extract_helvetica_bold():
    """macOS ships Helvetica as a collection (.ttc); write the Bold face to fonts/ as one file."""
    dst_dir = PROJ / 'fonts'
    for ttc in ('/System/Library/Fonts/Helvetica.ttc', '/Library/Fonts/Helvetica.ttc'):
        if not Path(ttc).exists():
            continue
        try:
            from fontTools.ttLib import TTCollection
            for f in TTCollection(ttc).fonts:
                full = (f['name'].getDebugName(4) or '').replace(' ', '').lower()
                if full == 'helveticabold':
                    dst_dir.mkdir(exist_ok=True)
                    dst = dst_dir / ('Helvetica-Bold.otf' if 'CFF ' in f else 'Helvetica-Bold.ttf')
                    if not dst.exists():
                        f.save(dst)
                    return dst
        except Exception as e:                                  # noqa: BLE001
            print(f'  (could not extract Helvetica Bold from {ttc}: {e})')
    return None


def setup_style():
    cands = []
    h = _extract_helvetica_bold()
    if h:
        cands.append(h)
    fdir = PROJ / 'fonts'
    if fdir.exists():                                  # any bold font file placed in fonts/ (e.g. Helvetica-Bold.ttf)
        cands += sorted(p for p in fdir.iterdir() if p.suffix.lower() in ('.ttf', '.otf') and 'bold' in p.name.lower())
    cands += [Path(p) for p in (
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf', '/Library/Fonts/Arial Bold.ttf',
        'C:/Windows/Fonts/arialbd.ttf', '/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf',
        '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf')]
    name, used = 'DejaVu Sans', None
    for p in cands:
        if p.exists():
            try:
                name, used = _try_font(p), p
                break
            except Exception:                                  # noqa: BLE001
                continue
    print(f'Font: {name} Bold ({used if used else "matplotlib default - install Helvetica or Arial"})')
    plt.rcParams.update({
        'font.family': name, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'font.size': FS['annot'], 'axes.labelsize': FS['label'], 'axes.titlesize': FS['label'],
        'xtick.labelsize': FS['tick'], 'ytick.labelsize': FS['tick'], 'legend.fontsize': FS['legend'],
        'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelpad': 3.0,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
        'lines.linewidth': 1.6, 'lines.markersize': 4, 'lines.solid_capstyle': 'round',
        'legend.frameon': False, 'legend.handlelength': 1.5, 'legend.handletextpad': 0.5,
        'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.3, 'legend.borderpad': 0.2,
        'errorbar.capsize': 0, 'axes.unicode_minus': True, 'axes.formatter.use_mathtext': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.dpi': 100, 'savefig.dpi': 600, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'hatch.linewidth': 0.8, 'axes.xmargin': 0.03, 'axes.ymargin': 0.05})


# ----------------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------------
def new_fig(mosaic, height, width=WIDTH, **gs):
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(width, height), layout='constrained', gridspec_kw=gs or None)
    fig.get_layout_engine().set(w_pad=3 / 72, h_pad=4 / 72, wspace=0.04, hspace=0.06)
    for key, ax in axd.items():
        if len(key) == 1 and key.isalpha():
            ax._letter = ax.set_title(key, loc='left', fontsize=FS['letter'], fontweight='bold', pad=7)
    return fig, axd


def _align_letters(fig):
    for _ in range(2):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax in fig.axes:
            t = getattr(ax, '_letter', None)
            if t is None:
                continue
            items = [ax.yaxis.label] + _ticklabels(ax, 'y')
            xs = [it.get_window_extent(r).x0 for it in items if it.get_visible() and it.get_text().strip()]
            x0 = min(xs + [ax.get_window_extent(r).x0])
            t.set_x((x0 - ax.bbox.x0) / ax.bbox.width)


def _ticklabels(ax, which):
    """Tick labels that are actually drawn (inside the axis limits)."""
    axis = ax.xaxis if which == 'x' else ax.yaxis
    lo, hi = sorted(axis.get_view_interval())
    eps = (hi - lo) * 1e-6
    out = []
    for tick in axis.get_major_ticks():
        if lo - eps <= tick.get_loc() <= hi + eps and tick.label1.get_visible():
            out.append(tick.label1)
    return out


def _texts(ax):
    out = [ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title]
    out += _ticklabels(ax, 'x') + _ticklabels(ax, 'y') + list(ax.texts)
    leg = ax.get_legend()
    if leg is not None:
        out += list(leg.get_texts())
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def check_layout(fig, name):
    """Warn about text that overlaps other text, another panel, or falls outside the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes = []
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        for t in _texts(ax):
            bb = t.get_window_extent(r)
            if bb.width > 0:
                boxes.append((ax, t, bb.padded(0.4)))
    issues = []
    for i in range(len(boxes)):
        a_ax, a_t, a = boxes[i]
        if a.x0 < fb.x0 - 1 or a.x1 > fb.x1 + 1 or a.y0 < fb.y0 - 1 or a.y1 > fb.y1 + 1:
            issues.append(f'"{a_t.get_text()}" is outside the figure')
        for j in range(i + 1, len(boxes)):
            b_ax, b_t, b = boxes[j]
            if a.overlaps(b):
                issues.append(f'"{a_t.get_text()}" overlaps "{b_t.get_text()}"')
        for ax in fig.axes:
            if ax is a_ax or not ax.axison:
                continue
            if a.overlaps(ax.bbox.shrunk(0.98, 0.98)):
                issues.append(f'"{a_t.get_text()}" runs into another panel')
    # legends / in-panel text on top of data
    for ax in fig.axes:
        if not ax.axison:
            continue
        pts, _, boxes = _data_points(ax, r)
        leg = ax.get_legend()
        if leg is not None:
            c = _covered(leg.get_window_extent(r), pts, boxes)
            if c:
                issues.append(f'legend covers data (panel {getattr(ax, "_letter", None) and ax._letter.get_text()}, score {c})')
        for t in ax.texts:
            if getattr(t, '_skip_check', False) or not t.get_text().strip():
                continue
            bb = t.get_window_extent(r)
            n = int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())
            if n >= 2:
                issues.append(f'"{t.get_text()}" covers data')
    for s in dict.fromkeys(issues):
        WARN.append(f'{name}: {s}')
        print(f'  WARNING {name}: {s}')
    return issues


def save(fig, name):
    _place_legends(fig)
    _align_letters(fig)
    check_layout(fig, name)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=300)
    try:
        fig.savefig(OUT / f'{name}.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    except Exception:                                          # noqa: BLE001
        pass
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f'  saved {name} ({w:.2f} x {h:.2f} in)')


def note_missing(ax, what):
    MISSING.append(what)
    ax.text(0.5, 0.5, f'missing:\n{what}', ha='center', va='center', transform=ax.transAxes,
            fontsize=7, color=NULL_L)
    ax.set_xticks([])
    ax.set_yticks([])


def _data_points(ax, r):
    """Display-space data points, reference-line points (zero/chance lines, brackets) and boxes (bars, texts)."""
    from matplotlib.collections import LineCollection, PathCollection
    pts, ref, boxes = [], [], []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        dest = ref if getattr(ln, '_skip_check', False) else pts
        xy = np.asarray(ln.get_xydata(), float)
        xy = xy[np.isfinite(xy).all(1)]
        if not len(xy):
            continue
        d = ln.get_transform().transform(xy)          # also right for axhline / axvline
        if len(d) > 1 and ln.get_linestyle() not in ('None', '', ' '):
            seg = [np.linspace(d[i], d[i + 1], max(2, int(np.hypot(*(d[i + 1] - d[i])) // 2))) for i in range(len(d) - 1)]
            d = np.concatenate(seg)
        dest.append(d)
    for coll in ax.collections:
        if getattr(coll, '_skip_check', False):
            continue
        if isinstance(coll, PathCollection):
            off = np.asarray(coll.get_offsets(), float)
            if len(off):
                pts.append(ax.transData.transform(off[np.isfinite(off).all(1)]))
        elif isinstance(coll, LineCollection):
            for sg in coll.get_segments():
                sg = np.asarray(sg, float)
                if len(sg) > 1 and np.isfinite(sg).all():
                    d = ax.transData.transform(sg)
                    pts.append(np.linspace(d[0], d[-1], 12))
    for pa in ax.patches:
        if getattr(pa, '_skip_check', False) or not pa.get_visible():
            continue
        bb = pa.get_window_extent(r)
        if bb.width > 0 and bb.height > 0:
            boxes.append(bb)
    for t in ax.texts:
        if t.get_visible() and t.get_text().strip():
            boxes.append(t.get_window_extent(r))
    cat = (lambda L: np.concatenate(L) if L else np.zeros((0, 2)))
    return cat(pts), cat(ref), boxes


def _inside(bb, pts):
    return int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())


def _covered(bb, pts, boxes):
    return _inside(bb, pts) + 50 * sum(bb.overlaps(b) for b in boxes)


def legend(*a, **k):
    """Legend placed later (in save) at the position that covers the least data."""
    ax = a[0]
    k.setdefault('fontsize', FS['legend'])
    k.update(frameon=True, facecolor='white', edgecolor='none', framealpha=1.0)
    ax._legend_req = (a[1:], k)
    return ax.legend(*a[1:], **k)


LOCS = ['upper left', 'upper right', 'lower right', 'lower left', 'center right', 'center left', 'upper center',
        'lower center']


def _place_legends(fig):
    for ax in fig.axes:
        req = getattr(ax, '_legend_req', None)
        if req is None:
            continue
        args, k = req
        pref = k.pop('loc', 'upper left')
        order = [pref] + [l_ for l_ in LOCS if l_ != pref]
        for attempt in range(4):
            best = None
            for loc in order:
                leg = ax.legend(*args, loc=loc, **k)
                leg.set_zorder(10)
                fig.canvas.draw()
                r = fig.canvas.get_renderer()
                pts, ref, boxes = _data_points(ax, r)
                bb = leg.get_window_extent(r)
                c = _covered(bb, pts, boxes)
                score = c + 0.02 * _inside(bb, ref)
                if best is None or score < best[0]:
                    best = (score, loc, c)
                if score == 0:
                    break
            if best[2] == 0:
                break
            lo, hi = ax.get_ylim()                     # make room at the top and try again
            ax.set_ylim(lo, hi + 0.18 * (hi - lo))
        leg = ax.legend(*args, loc=best[1], **k)
        leg.set_zorder(10)
        _trim_vlines(fig, ax, leg)


def _trim_vlines(fig, ax, leg):
    """Reference lines that run under a legend placed at a panel edge stop at the legend border,
    so no short stub sticks out between the legend and the axis edge."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = leg.get_window_extent(r).transformed(ax.transAxes.inverted())
    inv = ax.transAxes.inverted()
    for ln in ax.lines:
        if not getattr(ln, '_skip_check', False):
            continue
        xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(xd) != 2:
            continue
        if xd[0] == xd[1] and np.allclose(yd, [0, 1]):                      # vertical (axvline)
            xa = inv.transform(ax.transData.transform((xd[0], 0)))[0]
            if bb.x0 <= xa <= bb.x1:
                if bb.y1 > 0.88:
                    ln.set_ydata([0, bb.y0 - 0.02])
                elif bb.y0 < 0.12:
                    ln.set_ydata([bb.y1 + 0.02, 1])
        elif yd[0] == yd[1] and np.allclose(xd, [0, 1]):                    # horizontal (axhline)
            ya = inv.transform(ax.transData.transform((0, yd[0])))[1]
            if bb.y0 <= ya <= bb.y1:
                if bb.x1 > 0.88:
                    ln.set_xdata([0, bb.x0 - 0.01])
                elif bb.x0 < 0.12:
                    ln.set_xdata([bb.x1 + 0.01, 1])


def chance_line(ax, y, horizontal=True, **k):
    k = dict(color=NULL_L, ls=(0, (3, 2)), lw=0.9, zorder=0) | k
    ln = ax.axhline(y, **k) if horizontal else ax.axvline(y, **k)
    ln._skip_check = True
    return ln


def zero_vline(ax, x=0):
    ln = ax.axvline(x, color=INK, lw=0.8, zorder=0)
    ln._skip_check = True
    return ln


def stars(p):
    if not np.isfinite(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def bracket(ax, x1, x2, y, p, h=None, fs=None):
    h = h if h is not None else 0.025 * np.diff(ax.get_ylim())[0]
    ln, = ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=INK, lw=0.9, clip_on=False)
    ln._skip_check = True
    s = stars(p)
    t = ax.text((x1 + x2) / 2, y + h * (1.2 if s != 'n.s.' else 1.6), s, ha='center', va='bottom',
                fontsize=fs or (FS['annot'] + (1.5 if s != 'n.s.' else 0)))
    t._skip_check = True


def star_above(ax, x, y, p):
    t = ax.text(x, y, stars(p), ha='center', va='bottom', fontsize=FS['annot'] + (1.5 if stars(p) != 'n.s.' else 0))
    t._skip_check = True


def strip(ax, x, vals, color=DOT, width=0.16, size=11, rng=None, zorder=2, alpha=0.9, marker='o', face=None):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    rng = rng or np.random.default_rng(1)
    xs = x + rng.uniform(-width, width, len(vals))
    sc = ax.scatter(xs, vals, s=size, color=face or color, edgecolors=color, linewidths=0.6, zorder=zorder,
                    alpha=alpha, marker=marker)
    return sc


def mean_ci(ax, x, m, lo, hi, color=INK, ms=6, lw=1.6, marker='o', zorder=4, cap=0):
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt=marker, color=color, ms=ms, lw=lw, capsize=cap,
                mec=color, mfc=color, zorder=zorder)


def boot_ci(x, rng, n=2000, fn=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    b = [fn(rng.choice(x, len(x))) for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def sem_band(ax, x, m, s, color, alpha=0.2, **k):
    pc = ax.fill_between(x, m - s, m + s, color=color, alpha=alpha, lw=0, **k)
    pc._skip_check = True
    return pc


def wil(x, mu=0.0, alt='greater'):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - mu
    if len(x) < 5:
        return np.nan
    return stats.wilcoxon(x, alternative=alt).pvalue


def fmt_p(p):
    if not np.isfinite(p):
        return 'n/a'
    return 'P < 0.001' if p < 0.001 else f'P = {p:.3f}' if p < 0.01 else f'P = {p:.2f}'


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
def region_dirs():
    found = {}
    if (PROJ / 'results').exists():
        found['ZI'] = PROJ / 'results'
    for p in sorted(PROJ.glob('results_*')):
        reg = p.name.split('_', 1)[1]
        if reg != 'comparison' and p.is_dir():
            found[reg] = p
    order = [r for r in ORDER if r in found] + [r for r in found if r not in ORDER]
    return {r: found[r] for r in order}


REG = {}


def rcol(reg):
    if reg in REG_COL:
        return REG_COL[reg]
    extra = [r for r in REG if r not in REG_COL]
    return REG_EXTRA[extra.index(reg) % len(REG_EXTRA)] if reg in extra else NULL_L


def csv(path, note=True):
    path = Path(path)
    if not path.exists():
        if note:
            MISSING.append(str(path.relative_to(PROJ)))
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def per_region(fname):
    out = {}
    for reg, d in REG.items():
        df = csv(d / fname, note=(reg == 'ZI'))
        if df is not None and len(df):
            out[reg] = df
    return out


def parse_imposter(path):
    """real / imposter values from summary_07_imposter.txt."""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        m = re.match(r'\s*(.+?)\s*:\s*real ([\d.]+) \| imposter mean ([\d.]+) \(95th pct ([\d.]+)\) \| p = ([\d.eE+-]+)', line)
        if m:
            out[m.group(1).strip()] = dict(real=float(m.group(2)), imp=float(m.group(3)), p95=float(m.group(4)),
                                           p=float(m.group(5)))
    return out


def parse_checks(path):
    """Mouse-clustered unit-fraction tests and the within-insertion overlap test from summary_15."""
    out = {}
    if not Path(path).exists():
        return out
    kind = None
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip().startswith('direction units'):
            kind = 'direction'
        elif line.strip().startswith('speed units'):
            kind = 'speed'
        m = re.match(r'\s+ZI - (\S+): [+-][\d.]+ points \(mouse-bootstrap 95% CI .*?; p=([\d.eE+-]+)\)', line)
        if m and kind:
            out[(kind, m.group(1))] = float(m.group(2))
        m = re.match(r'\s+both (\d+) .*expected ([\d.]+) .*p=([\d.eE+-]+)', line)
        if m:
            out['overlap'] = dict(both=int(m.group(1)), expected=float(m.group(2)), p=float(m.group(3)))
    return out


def parse_pose_latency(path):
    out, reg = {}, None
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        m = re.match(r'--- (\S+) \((\d+) insertions\)', line)
        if m:
            reg = m.group(1)
            out[reg] = dict(n=int(m.group(2)))
            continue
        m = re.search(r'half-max latency: ([+-]?\d+) ms \(paired bootstrap 95% CI ([+-]?\d+) to ([+-]?\d+); p=([\d.eE+-]+)', line)
        if m and reg:
            out[reg].update(diff=float(m.group(1)), lo=float(m.group(2)), hi=float(m.group(3)), p=float(m.group(4)))
            continue
        m = re.match(r'\s+(neural - pose|neural|pose only)\s*: half-max ([+-]?\d+|nan) ms', line)
        if m and reg:
            out[reg][m.group(1)] = float(m.group(2))
    return out


def zi_sessions():
    paths = sorted(p for p in (PROJ / 'processed').iterdir() if (p / 'meta.json').exists()) \
        if (PROJ / 'processed').exists() else []
    return paths


def right_choice_code(t):
    rc = np.sign(t.loc[t['signed_contrast'] > 0, 'choice'].mean())
    return rc if rc != 0 and np.isfinite(rc) else -1.0


_RC = []


def majority_right_code():
    """IBL choice code (+1 or -1) that reports a right-side stimulus in most ZI sessions."""
    if not _RC:
        codes = []
        for p_ in zi_sessions():
            _, t, _, _ = K.load_insertion(p_)
            codes.append(right_choice_code(t))
        _RC.append(1.0 if (np.array(codes) == 1).mean() > 0.5 else -1.0)
    return _RC[0]


def glm_units_locations():
    out = {}
    for reg, d in REG.items():
        g = csv(d / 'glm_units.csv', note=False)
        if g is not None:
            u = g.drop_duplicates(['pid', 'unit_index'])[['x', 'y', 'z']] * 1e3
            out[reg] = u
    return out


def add_leg(fig, text):
    LEGENDS.setdefault(fig, []).append(text)


# ============================================================================
# Figure 1 - task, recordings, behaviour, choice-movement confound
# ============================================================================
def figure1():
    fig, A = new_fig([['A', 'B', 'C'], ['D', 'E', 'F']], 4.7)
    locs = glm_units_locations()
    for key, (cx, xl) in zip('AB', (('y', 'AP from bregma (mm)'), ('x', 'ML from bregma (mm)'))):
        ax = A[key]
        if not locs:
            note_missing(ax, 'results*/glm_units.csv')
            continue
        for reg, u in locs.items():
            sc = ax.scatter(u[cx], u['z'], s=5, color=rcol(reg), lw=0, alpha=0.55, rasterized=True)
            sc._skip_check = True
        ax.set(xlabel=xl, ylabel='DV (mm)')
        allx = np.concatenate([u[cx].to_numpy() for u in locs.values()])
        ally = np.concatenate([u['z'].to_numpy() for u in locs.values()])
        xr, yr = np.ptp(allx) or 1, np.ptp(ally) or 1
        ax.set_xlim(allx.min() - 0.12 * xr, allx.max() + 0.12 * xr)
        ax.set_ylim(ally.min() - 0.2 * yr, ally.max() + 0.2 * yr)
        for reg, u in locs.items():
            ux, uy = u[cx].to_numpy(), u['z'].to_numpy()
            cands = [(np.median(ux), np.percentile(uy, 95) + 0.06 * yr, 'center', 'bottom'),
                     (np.median(ux), np.percentile(uy, 5) - 0.06 * yr, 'center', 'top'),
                     (np.percentile(ux, 90) + 0.04 * xr, np.median(uy), 'left', 'center'),
                     (np.percentile(ux, 10) - 0.04 * xr, np.median(uy), 'right', 'center')]
            best, score = None, None
            for x_, y_, ha, va in cands:
                w, h = 0.11 * xr, 0.07 * yr
                x0 = x_ - w / 2 if ha == 'center' else (x_ if ha == 'left' else x_ - w)
                y0 = y_ - h / 2 if va == 'center' else (y_ if va == 'bottom' else y_ - h)
                inside = ((allx > x0) & (allx < x0 + w) & (ally > y0) & (ally < y0 + h)).sum()
                outside = not (ax.get_xlim()[0] < x0 and x0 + w < ax.get_xlim()[1]
                               and ax.get_ylim()[0] < y0 and y0 + h < ax.get_ylim()[1])
                sc_ = inside + 1000 * outside
                if score is None or sc_ < score:
                    best, score = (x_, y_, ha, va), sc_
            ax.text(best[0], best[1], reg, color=rcol(reg), ha=best[2], va=best[3], fontsize=FS['annot'] + 0.5)
        add_leg('Fig1', f'{key}: recorded units ({", ".join(f"{r} n = {len(u)}" for r, u in locs.items())}) in '
                        f'{"sagittal" if key == "A" else "coronal"} projection (IBL coordinates relative to bregma).')

    paths = zi_sessions()
    psy, rts, wv_r, wv_l = [], [], [], []
    edges, cent = K.bin_centers()
    for p in paths:
        meta, t, u, b = K.load_insertion(p)
        rc = right_choice_code(t)
        tt = t[t['choice'] != 0]
        for blk, g in tt.groupby(np.round(tt['probabilityLeft'], 1)):
            for c, gg in g.groupby('signed_contrast'):
                psy.append(dict(pid=meta['pid'], block=blk, contrast=c, p_right=(gg['choice'] == rc).mean()))
        rts.append(t['rt'].to_numpy())
        ok = K.valid_trials(t)
        if 'wheel_vel_fm' in b:
            w = b['wheel_vel_fm'][ok]
            ch = t.loc[ok, 'choice'].to_numpy()
            if (ch == rc).sum() > 5 and (ch == -rc).sum() > 5:
                wv_r.append(w[ch == rc].mean(0))
                wv_l.append(w[ch == -rc].mean(0))

    ax = A['C']
    if psy:
        psy = pd.DataFrame(psy)
        cons = np.sort(psy['contrast'].unique())
        pos = {c: i for i, c in enumerate(cons)}
        for blk, col, lab in ((0.2, DIR, 'Right block'), (0.5, '#555555', 'Unbiased'), (0.8, LEFT, 'Left block')):
            g = psy[psy['block'] == blk]
            if g.empty:
                continue
            m = g.groupby('contrast')['p_right'].agg(['mean', K.sem]).reindex(cons)
            x = np.array([pos[c] for c in m.index])
            ax.errorbar(x, m['mean'], yerr=m['sem'], color=col, marker='o', ms=3.5, lw=1.4, label=lab,
                        ls='-' if blk != 0.5 else (0, (3, 1.5)))
        ax.set_xticks([pos[c] for c in cons])
        ax.set_xticklabels([f'{c * 100:.0f}'.replace('-', '\u2212') if (np.isclose(abs(c), 1.0) or c == 0) else ''
                            for c in cons])
        ax.set(xlabel='Signed contrast (%)', ylabel='P(rightward choice)', ylim=(-0.02, 1.02), yticks=[0, 0.5, 1])
        legend(ax, loc='upper left', handlelength=1.2)
        add_leg('Fig1', f'C: psychometric curves of the ZI sessions by block (mean ± SEM across {psy["pid"].nunique()} sessions); '
                        f'contrast is right minus left.')
    else:
        note_missing(ax, 'processed/ (ZI)')

    ax = A['D']
    if rts:
        rt = np.concatenate(rts) * 1000
        rt = rt[np.isfinite(rt)]
        ax.hist(rt[(rt > 0) & (rt < 1000)], bins=np.arange(0, 1001, 20), color='#9A9A9A', lw=0)
        med = np.median(rt)
        ln = ax.axvline(med, color=INK, lw=1.0, ls=(0, (3, 2)))
        ln._skip_check = True
        t = ax.text(med + 45, ax.get_ylim()[1] * 0.95, f'Median\n{med:.0f} ms', va='top', ha='left')
        t._skip_check = True
        ax.set(xlabel='Reaction time (ms)', ylabel='Trials', xlim=(0, 1000))
        add_leg('Fig1', f'D: time from stimulus onset to wheel movement onset, all ZI trials (n = {len(rt)}); median {med:.0f} ms.')

    ax = A['E']
    if wv_r:
        cm = cent * 1000
        sel = (cm >= -200) & (cm <= 400)
        for arr, col, lab in ((np.array(wv_r), DIR, 'Rightward'), (np.array(wv_l), LEFT, 'Leftward')):
            m, s = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(len(arr))
            ax.plot(cm[sel], m[sel], color=col, label=lab)
            sem_band(ax, cm[sel], m[sel], s[sel], col)
        zero_vline(ax)
        chance_line(ax, 0, color=INK, ls='-', lw=0.6)
        ax.set(xlabel='Time from movement (ms)', ylabel='Wheel velocity (rad/s)', xlim=(-200, 400))
        legend(ax, loc='upper right')
        add_leg('Fig1', f'E: wheel velocity aligned to detected movement onset, mean ± SEM across {len(wv_r)} sessions.')

    ax = A['F']
    conf = csv(PROJ / 'results' / 'choice_speed_confound.csv')
    if conf is not None:
        cs = conf.sort_values('cohen_d_logspeed').reset_index(drop=True)
        sig = cs['p'] < 0.05
        x = np.arange(len(cs))
        ax.scatter(x[sig], cs.loc[sig, 'cohen_d_logspeed'], s=16, color=VIG, lw=0, label='P < 0.05', zorder=3)
        ax.scatter(x[~sig], cs.loc[~sig, 'cohen_d_logspeed'], s=16, facecolor='white', edgecolor=NULL_L, lw=0.8,
                   label='n.s.', zorder=3)
        chance_line(ax, 0, color=INK, ls='-', lw=0.6)
        ax.set(xlabel='Insertion (sorted)', ylabel='Speed difference (d)', xticks=[0, len(cs) - 1],
               xticklabels=['1', str(len(cs))])
        legend(ax, loc='upper left', handletextpad=0.2, markerscale=1.1)
        add_leg('Fig1', f'F: peak wheel speed differs between the two choices in {int(sig.sum())}/{len(cs)} insertions '
                        f'(Mann-Whitney P < 0.05; median |d| = {cs["cohen_d_logspeed"].abs().median():.2f}): '
                        f'choice and movement vigour are confounded.')
    save(fig, 'Fig1')


# ============================================================================
# Figure 2 - single units: movement-dominated encoding with direction units
# ============================================================================
TERM_LAB = {'choice': 'Direction', 'stimulus': 'Stimulus', 'prior': 'Block prior', 'speed': 'Speed',
            'rt': 'Reaction time', 'dur': 'Duration', 'me': 'Video motion'}
TERM_COL = {'choice': DIR, 'stimulus': STIM, 'prior': PRIOR, 'speed': VIG, 'rt': '#7F7F7F', 'dur': '#7F7F7F',
            'me': '#7F7F7F'}


def _example_unit(g, window, term, min_rate=2.0):
    g = g[(g['window'] == window) & (g['term'] == term)].sort_values('F', ascending=False)
    for _, best in g.head(100).iterrows():
        path = PROJ / 'processed' / best['pid']
        if not path.exists():
            continue
        meta, t, u, b = K.load_insertion(path)
        ok = K.valid_trials(t)
        hit = np.where(u['unit_index'].to_numpy() == best['unit_index'])[0]
        if not len(hit):
            continue
        cc = b['fm_counts'][hit[0]][ok].astype(float)            # trials x bins
        if cc.mean() / C.BIN >= min_rate and ok.sum() >= 100:
            return cc, t[ok].reset_index(drop=True), best
    return None


def _psth(ax, counts, sel, col, label, cm, sl, sigma=1.5, band=True, **k):
    r = gaussian_filter1d(counts[sel] / C.BIN, sigma, axis=1)
    m, s = r.mean(0), r.std(0, ddof=1) / np.sqrt(sel.sum())
    ax.plot(cm[sl], m[sl], color=col, label=label, **k)
    if band:
        sem_band(ax, cm[sl], m[sl], s[sl], col, alpha=0.18)


def window_band(ax, a, b):
    sp = ax.axvspan(a, b, color='#EFEFEF', lw=0, zorder=-1)
    sp._skip_check = True


def figure2():
    fig, A = new_fig([['A', 'B', 'C'], ['D', 'E', 'F']], 4.9, width_ratios=[1, 1, 1.12])
    g = csv(PROJ / 'results' / 'glm_units.csv')
    summ = csv(PROJ / 'results' / 'glm_summary.csv')
    tc = csv(PROJ / 'results' / 'glm_timecourse.csv')
    _, cent = K.bin_centers()
    cm = cent * 1000
    sl = (cm >= -300) & (cm <= 300)

    ex = _example_unit(g, 'premove', 'choice') if g is not None else None
    ax = A['A']
    if ex:
        cc, t, best = ex
        rc = right_choice_code(t)
        ch = t['choice'].to_numpy()
        window_band(ax, -160, 0)
        _psth(ax, cc, ch == rc, DIR, 'Rightward', cm, sl)
        _psth(ax, cc, ch == -rc, LEFT, 'Leftward', cm, sl)
        zero_vline(ax)
        ax.set(xlabel='Time from movement (ms)', ylabel='Firing rate (spikes/s)', xlim=(-300, 300))
        ax.set_ylim(bottom=0)
        legend(ax, loc='upper left')
        add_leg('Fig2', f'A: example direction unit (insertion {best["pid"][:8]}, unit {int(best["unit_index"])}; '
                        f'premove direction F = {best["F"]:.1f}, permutation P = {best["p_perm"]:.4f}). Shaded: '
                        f'pre-movement window (−160 to 0 ms). Mean ± SEM across trials, Gaussian smoothing σ = 30 ms.')
    else:
        note_missing(ax, 'example direction unit (processed/, glm_units.csv)')

    ex = _example_unit(g, 'move', 'speed') if g is not None else None
    ax = A['B']
    if ex:
        cc, t, best = ex
        terc = pd.qcut(t['peak_speed'], 3, labels=False).to_numpy()
        window_band(ax, 0, 200)
        for k_, (col, lab) in enumerate(zip(VIG_SH, ('Slow', 'Medium', 'Fast'))):
            _psth(ax, cc, terc == k_, col, lab, cm, sl)
        zero_vline(ax)
        ax.set(xlabel='Time from movement (ms)', ylabel='Firing rate (spikes/s)', xlim=(-300, 300))
        ax.set_ylim(bottom=0)
        legend(ax, loc='upper left', title=None)
        add_leg('Fig2', f'B: example speed unit (insertion {best["pid"][:8]}, unit {int(best["unit_index"])}), trials split '
                        f'into terciles of peak wheel speed. Shaded: movement window (0-200 ms).')
    else:
        note_missing(ax, 'example speed unit')

    ax = A['C']
    if summ is not None:
        terms = [t_ for t_ in ('choice', 'stimulus', 'prior', 'speed', 'rt', 'dur', 'me') if t_ in summ['term'].values]
        y = np.arange(len(terms))[::-1]
        hgt = 0.38
        for off, w, alpha, hatch in ((hgt / 2, 'premove', 1.0, None), (-hgt / 2, 'move', 0.45, None)):
            s = summ[summ['window'] == w].set_index('term').reindex(terms)
            ax.barh(y + off, s['frac_sig'], hgt, color=[TERM_COL[t_] for t_ in terms], alpha=alpha, lw=0)
            ax.errorbar(s['frac_sig'], y + off, xerr=[s['frac_sig'] - s['ci_low'], s['ci_high'] - s['frac_sig']],
                        fmt='none', ecolor=INK, lw=0.8)
        chance_line(ax, 0.05, horizontal=False)
        ax.set_yticks(y)
        ax.set_yticklabels([TERM_LAB[t_] for t_ in terms])
        ax.set(xlabel='Fraction of ZI units', xlim=(0, None))
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        h = [Patch(color='#555555', label='Pre-movement'), Patch(color='#555555', alpha=0.45, label='Movement')]
        legend(ax, handles=h, loc='lower right', handlelength=1.0)
        pm = summ[(summ['window'] == 'premove') & (summ['term'] == 'choice')].iloc[0]
        mv = summ[(summ['window'] == 'move') & (summ['term'] == 'speed')].iloc[0]
        add_leg('Fig2', f'C: fraction of ZI units (n = {int(pm["n_units"])}) with a significant effect of each regressor '
                        f'(Freedman-Lane permutation, P < 0.05; error bars, 95% binomial CI; dashed line, 5%). '
                        f'Direction, pre-movement: {pm["frac_sig"]:.1%} ({int(pm["n_sig"])} units; FDR {int(pm["n_sig_fdr"])}); '
                        f'speed, movement: {mv["frac_sig"]:.1%}.')
    else:
        note_missing(ax, 'results/glm_summary.csv')

    ax = A['D']
    if tc is not None:
        for term, col, lab in (('choice', DIR, 'Direction'), ('speed', VIG, 'Speed'), ('prior', PRIOR, 'Prior')):
            s = tc[tc['term'] == term].sort_values('t_center')
            ax.plot(s['t_center'] * 1000, s['frac'], color=col, label=lab, marker='o', ms=2.5)
        chance_line(ax, 0.05)
        zero_vline(ax)
        ax.set(xlabel='Time from movement (ms)', ylabel='Fraction of units', ylim=(0, None))
        legend(ax, loc='upper left')
        add_leg('Fig2', 'D: time-resolved encoding (100-ms windows, 40-ms steps; parametric F-test, P < 0.05).')
    else:
        note_missing(ax, 'results/glm_timecourse.csv')

    ax = A['E']
    if g is not None:
        pm = g[g['window'] == 'premove'].pivot_table(index=['pid', 'unit_index'], columns='term', values='p_perm')
        sc_, ss_ = pm['choice'] < C.ALPHA, pm['speed'] < C.ALPHA
        both, c_only, s_only = int((sc_ & ss_).sum()), int((sc_ & ~ss_).sum()), int((~sc_ & ss_).sum())
        neither = int((~sc_ & ~ss_).sum())
        tot = both + c_only + s_only + neither
        chk = parse_checks(CMP / 'summary_15_revision_checks.txt').get('overlap')
        vals = np.array([c_only, both, s_only]) / tot
        ax.bar([0, 1, 2], vals, 0.62, color=[DIR, MIX, VIG], lw=0)
        if chk:                                     # expected overlap under within-insertion shuffling
            exp_both, test_txt = chk['expected'] / tot, f'within-insertion permutation test, {fmt_p(chk["p"])}'
        else:
            odds, pf = stats.fisher_exact([[both, c_only], [s_only, neither]])
            exp_both, test_txt = (c_only + both) * (s_only + both) / tot ** 2, f'Fisher exact test OR = {odds:.2f}, {fmt_p(pf)}'
        exp_lab = 'expected under within-insertion shuffling' if chk else 'expected if independent'
        ln, = ax.plot([0.62, 1.38], [exp_both] * 2, color=INK, lw=1.1, ls=(0, (2, 1.5)))
        ln._skip_check = True
        for xi, n_ in enumerate((c_only, both, s_only)):
            t = ax.text(xi, vals[xi] + 0.006, str(n_), ha='center', va='bottom')
            t._skip_check = True
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Direction\nonly', 'Both', 'Speed\nonly'])
        ax.set(ylabel='Fraction of units', ylim=(0, max(vals) * 1.3))
        add_leg('Fig2', f'E: overlap of direction and speed coding before movement (counts above bars; dashed line, '
                        f'overlap {exp_lab}). {test_txt}.')
    else:
        note_missing(ax, 'results/glm_units.csv')

    ax = A['F']
    null = csv(PROJ / 'results' / 'imposter_null.csv')
    imp = parse_imposter(PROJ / 'results' / 'summary_07_imposter.txt')
    if null is not None:
        items = [('premove_choice', 'Choice units (premove)', 'Direction\n(pre-move)', DIR),
                 ('move_speed', 'Speed units (move)', 'Speed\n(move)', VIG),
                 ('iti_prior', 'Prior units (iti)', 'Prior\n(ITI)', PRIOR)]
        txt = []
        for xi, (col, key, lab, colr) in enumerate(items):
            v = null[col].dropna().to_numpy()
            vp = ax.violinplot(v, [xi], widths=0.55, showextrema=False)
            for b_ in vp['bodies']:
                b_.set_facecolor(NULL_F)
                b_.set_edgecolor('none')
                b_.set_alpha(1)
                b_._skip_check = True
            real = imp.get(key, {}).get('real', np.nan)
            ax.plot([xi], [real], marker='D', color=colr, ms=6, ls='none', zorder=4)
            p = imp.get(key, {}).get('p', (1 + (v >= real).sum()) / (1 + len(v)) if np.isfinite(real) else np.nan)
            star_above(ax, xi, max(real, v.max()) + 0.012, p)
            txt.append(f'{lab.replace(chr(10), " ")}: real {real:.3f} vs imposter {v.mean():.3f}, P = {p:.3g}')
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([it[2] for it in items])
        ax.set(ylabel='Fraction of units', ylim=(0, None), xlim=(-0.6, len(items) - 0.4))
        h = [Patch(color=NULL_F, label='Imposter'), Line2D([], [], marker='D', color=INK, ls='none', ms=5, label='Real')]
        legend(ax, handles=h, loc='upper right', handletextpad=0.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        add_leg('Fig2', f'F: real fractions (diamonds) against an imposter-session null in which neural data are paired '
                        f'with behaviour from other sessions ({len(null)} pairings; grey, null distribution). ' + '; '.join(txt) + '.')
    else:
        note_missing(ax, 'results/imposter_null.csv')
    save(fig, 'Fig2')


# ============================================================================
# Figure 3 - population decoding with movement controls
# ============================================================================
def paired(ax, a, b, labels, cols=(NULL_L, DIR), rng=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    for ai, bi in zip(a[ok], b[ok]):
        ln, = ax.plot([0, 1], [ai, bi], color='#C8C8C8', lw=0.7, zorder=1)
        ln._skip_check = True
    for xi, v, c in ((0, a[ok], cols[0]), (1, b[ok], cols[1])):
        sc = ax.scatter(np.full(ok.sum(), xi), v, s=10, color=c, lw=0, alpha=0.65, zorder=2)
        sc._skip_check = True
        mean_ci(ax, xi + (-0.22 if xi == 0 else 0.22), v.mean(), v.mean() - K.sem(v), v.mean() + K.sem(v), color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, 1.5)


def figure3():
    fig, A = new_fig([['A', 'B', 'C'], ['D', 'E', 'F']], 4.9)
    d = csv(PROJ / 'results' / 'decoding_sessions.csv')
    null = csv(PROJ / 'results' / 'imposter_null.csv')
    imp = parse_imposter(PROJ / 'results' / 'summary_07_imposter.txt')
    ch_m = null['acc_matched'].mean() if null is not None else 0.5
    ch_u = null['acc_unmatched'].mean() if null is not None else 0.5

    ax = A['A']
    if d is not None:
        paired(ax, d['acc_unmatched'], d['acc_matched'], ['All\ntrials', 'Speed-\nmatched'])
        for xi, c in ((0, ch_u), (1, ch_m)):
            ln, = ax.plot([xi - 0.32, xi + 0.32], [c, c], color=NULL_L, lw=1.2, ls=(0, (3, 2)))
            ln._skip_check = True
        ax.set(ylabel='Decoding accuracy')
        top = np.nanmax(d[['acc_unmatched', 'acc_matched']].to_numpy())
        for xi, col, c in ((0, 'acc_unmatched', ch_u), (1, 'acc_matched', ch_m)):
            star_above(ax, xi, top + 0.015, wil(d[col], c))
        ax.set_ylim(min(0.4, np.nanmin(d[['acc_unmatched', 'acc_matched']].to_numpy()) - 0.02), top + 0.08)
        mouse = d.groupby('subject')['acc_matched'].mean()
        add_leg('Fig3', f'A: direction decoding from pre-movement population activity (−160 to 0 ms; L2 logistic regression, '
                        f'balanced accuracy, 5-fold CV) in {len(d)} insertions ({d["subject"].nunique()} mice). Speed-matched: '
                        f'trials subsampled so that both choices are equally represented in each speed quintile. Dashed: '
                        f'imposter-session chance ({ch_u:.3f} / {ch_m:.3f}). Matched {d["acc_matched"].mean():.3f} ± '
                        f'{K.sem(d["acc_matched"]):.3f}, Wilcoxon vs chance {fmt_p(wil(d["acc_matched"], ch_m))}; mouse level '
                        f'{fmt_p(wil(mouse, ch_m))}. Black: mean ± SEM.')
    else:
        note_missing(ax, 'results/decoding_sessions.csv')

    ax = A['B']
    if null is not None:
        v = null['acc_matched'].dropna().to_numpy()
        real = imp.get('Choice decoding (matched)', {}).get('real', d['acc_matched'].mean() if d is not None else np.nan)
        ax.hist(v, bins=18, color=NULL_F, lw=0)
        ln = ax.axvline(real, color=DIR, lw=2.0)
        ln._skip_check = True
        p = imp.get('Choice decoding (matched)', {}).get('p', (1 + (v >= real).sum()) / (1 + len(v)))
        ymax = ax.get_ylim()[1]
        t = ax.text(real - 0.004, ymax * 1.02, 'Real', color=DIR, ha='right', va='bottom')
        t._skip_check = True
        t = ax.text(np.mean(v), ymax * 1.02, 'Imposter', color=NULL_L, ha='center', va='bottom')
        t._skip_check = True
        lo = min(v.min(), real)
        hi = max(v.max(), real)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel='Mean decoding accuracy', ylabel='Pairings', xlim=(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo)),
               ylim=(0, ymax * 1.18))
        add_leg('Fig3', f'B: mean speed-matched accuracy over insertions (blue, {real:.3f}) against {len(v)} imposter '
                        f'pairings (grey; mean {v.mean():.3f}); permutation P = {p:.3g}.')
    else:
        note_missing(ax, 'results/imposter_null.csv')

    ax = A['C']
    if d is not None:
        rng = np.random.default_rng(3)
        txt = []
        for xi, (col, lab) in enumerate((('acc_corr_to_error', 'Error'), ('acc_corr_to_zero', '0% contrast'))):
            v = d[col].dropna()
            strip(ax, xi, v, color=DIR, face='white', rng=rng)
            mean_ci(ax, xi + 0.3, v.mean(), v.mean() - K.sem(v), v.mean() + K.sem(v))
            p = wil(v, 0.5)
            star_above(ax, xi, v.max() + 0.015, p)
            txt.append(f'{lab} {v.mean():.3f} (n = {len(v)}, {fmt_p(p)})')
        chance_line(ax, 0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Error\ntrials', '0%\ncontrast'])
        ax.set(ylabel='Decoding accuracy', xlim=(-0.5, 1.6))
        ax.set_ylim(top=ax.get_ylim()[1] + 0.03)
        add_leg('Fig3', 'C: decoders trained on correct, non-zero-contrast trials (speed-matched) and tested on error '
                        'trials or 0% contrast trials. A stimulus decoder would fall below 0.5 on errors. ' + '; '.join(txt) + '.')

    ax = A['D']
    if d is not None:
        paired(ax, d['acc_within_speed'], d['acc_cross_speed'], ['Within\nspeed', 'Cross-\nspeed'])
        chance_line(ax, 0.5)
        ax.set(ylabel='Decoding accuracy')
        top = np.nanmax(d[['acc_within_speed', 'acc_cross_speed']].to_numpy())
        p = stats.wilcoxon(d['acc_within_speed'], d['acc_cross_speed']).pvalue
        bracket(ax, 0, 1, top + 0.02, p, h=0.012)
        ax.set_ylim(top=top + 0.08)
        add_leg('Fig3', f'D: decoders trained on fast trials tested on slow trials and vice versa (cross) versus within the same '
                        f'speed half: {d["acc_within_speed"].mean():.3f} vs {d["acc_cross_speed"].mean():.3f}, paired '
                        f'Wilcoxon {fmt_p(p)}; cross vs 0.5 {fmt_p(wil(d["acc_cross_speed"], 0.5))}.')

    ax = A['E']
    f = PROJ / 'results' / 'onset_sessions.npz'
    if f.exists():
        z = np.load(f)
        cms, real, impm = z['centers'] * 1000, z['real'], z['imp_mean']
        for arr, col, lab in ((impm, NULL_L, 'Imposter'), (real, DIR, 'Real')):
            m, s = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(len(arr))
            ax.plot(cms, m, color=col, label=lab)
            sem_band(ax, cms, m, s, col)
        ht = csv(PROJ / 'results' / 'history_timecourse.csv', note=False)
        if ht is not None and 'q' in ht:
            sig = ht['q'].to_numpy() < C.ALPHA
            ytop = (real.mean(0) + real.std(0, ddof=1) / np.sqrt(len(real))).max() + 0.012
            step = np.median(np.diff(cms)) if len(cms) > 1 else 40
            for xc in cms[sig]:
                ax.plot([xc - step / 2, xc + step / 2], [ytop, ytop], color=INK, lw=2.2, solid_capstyle='butt')
        zero_vline(ax)
        ax.set(xlabel='Time from movement (ms)', ylabel='Decoding accuracy')
        legend(ax, loc='upper left')
        ax.set_ylim(top=ax.get_ylim()[1] + 0.01)
        add_leg('Fig3', f'E: time-resolved decoding (sliding windows), matched for speed, block and previous choice, versus '
                        f'imposter sessions ({real.shape[0]} insertions; mean ± SEM). Black bar: windows where real > imposter '
                        f'(Wilcoxon, FDR q < 0.05).')
    else:
        note_missing(ax, 'results/onset_sessions.npz')

    ax = A['F']
    hg = csv(PROJ / 'results' / 'history_glm.csv')
    if hg is not None:
        hg = hg[hg['term'] == 'choice'].copy()
        hg['frac'] = hg['n_sig'] / hg['n_units']
        xx = np.arange(2)
        vals = {}
        for off, model, col, lab in ((-0.2, 'base', '#9A9A9A', 'Base model'), (0.2, 'history', DIR, '+ Trial history')):
            v = [hg[(hg['window'] == w) & (hg['model'] == model)]['frac'].iloc[0] for w in ('iti', 'premove')]
            vals[model] = v
            ax.bar(xx + off, v, 0.38, color=col, label=lab, lw=0)
        chance_line(ax, 0.05)
        ax.set_xticks(xx)
        ax.set_xticklabels(['ITI', 'Pre-\nmovement'])
        ax.set(ylabel='Direction units (fraction)', ylim=(0, max(max(vals['base']), max(vals['history'])) * 1.35))
        legend(ax, loc='upper left')
        add_leg('Fig3', f'F: adding previous choice, previous outcome and their interaction to the GLM removes most '
                        f'ITI "choice" units ({vals["base"][0]:.1%} → {vals["history"][0]:.1%}) but not pre-movement direction '
                        f'units ({vals["base"][1]:.1%} → {vals["history"][1]:.1%}); parametric P < 0.05; dashed line, 5%.')
    else:
        note_missing(ax, 'results/history_glm.csv')
    save(fig, 'Fig3')


# ============================================================================
# Figure 4 - action direction, not abstract choice: change of mind and body timing
# ============================================================================
def pose_tc():
    f = CMP / 'pose_timecourse.csv'
    return csv(f)


def figure4():
    fig, A = new_fig([['A', 'B', 'C'], ['D', 'E', 'F']], 4.9)
    rng = np.random.default_rng(4)
    rc = csv(CMP / 'region_comparison.csv')

    ax = A['A']
    com = per_region('followup_com_trials.csv')
    if com:
        txt = []
        for xi, (reg, c) in enumerate(com.items()):
            c = c[c['window'] == 'premove'] if 'window' in c else c
            per = c.groupby('pid')['agree_initial'].agg(['mean', 'size'])
            per = per[per['size'] >= 3]['mean']
            strip(ax, xi, per, color=rcol(reg), face='white', rng=rng, size=10)
            pids = c['pid'].unique()
            grp = {p_: c.loc[c['pid'] == p_, 'agree_initial'].to_numpy() for p_ in pids}
            bs = [np.mean(np.concatenate([grp[p_] for p_ in rng.choice(pids, len(pids))])) for _ in range(2000)]
            m = c['agree_initial'].mean()
            lo, hi = np.percentile(bs, [2.5, 97.5])
            mean_ci(ax, xi + 0.3, m, lo, hi, color=rcol(reg))
            p = stats.wilcoxon(per - 0.5).pvalue if len(per) >= 5 else np.nan
            star_above(ax, xi, min(per.max(), 1.0) + 0.03, p)
            txt.append(f'{reg} {m:.3f} [{lo:.3f}, {hi:.3f}] ({len(c)} trials, {len(per)} sessions, Wilcoxon {fmt_p(p)})')
        chance_line(ax, 0.5)
        ax.set_xticks(range(len(com)))
        ax.set_xticklabels(list(com))
        ax.set(ylabel='P(initial direction)', xlim=(-0.5, len(com) - 0.3), ylim=(0, 1.12), yticks=[0, 0.5, 1])
        add_leg('Fig4', 'A: change-of-mind trials (initial wheel direction opposite to the final choice). A decoder trained '
                        'on consistent trials (pre-movement window) reports the INITIAL movement direction more often than the '
                        'final choice. Open circles, sessions (≥ 3 trials); filled, pooled mean with session-bootstrap 95% CI: '
                        + '; '.join(txt) + '.')
    else:
        note_missing(ax, 'followup_com_trials.csv')

    ax = A['B']
    rev = per_region('reversal_curves.csv')
    if rev:
        for reg, cv in rev.items():
            taus = np.array([float(c_.split('_')[1].replace('ms', '')) for c_ in cv.columns])
            arr = cv.to_numpy(float)
            m = np.nanmean(arr, 0)
            s = np.nanstd(arr, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(arr), 0))
            main = reg == 'ZI'
            ax.plot(taus, m, color=rcol(reg), lw=1.9 if main else 1.2, label=f'{reg} (n = {len(arr)})', zorder=3 if main else 2)
            if main:
                sem_band(ax, taus, m, s, rcol(reg))
        if rc is not None and 'reversal_cross' in rc:
            r = rc.set_index('region')
            if 'ZI' in r.index and np.isfinite(r.loc['ZI', 'reversal_cross']):
                x0, lo, hi = r.loc['ZI', ['reversal_cross', 'rev_lo', 'rev_hi']]
                ax.errorbar([x0], [0.5], xerr=[[x0 - lo], [hi - x0]], fmt='o', color=DIR, ms=5, lw=1.6, capsize=2.5, zorder=5)
        chance_line(ax, 0.5)
        zero_vline(ax)
        ax.set(xlabel='Time from wheel reversal (ms)', ylabel='P(decoded = final choice)', ylim=(0.2, 0.8))
        legend(ax, loc='upper left')
        add_leg('Fig4', 'B: decoder output on change-of-mind trials aligned to the wheel reversal (mean ± SEM across sessions '
                        'for ZI; mean for other regions). Point: time at which the ZI decoder switches to the final '
                        'direction, with bootstrap 95% CI.')
    else:
        note_missing(ax, 'reversal_curves.csv')

    pt = pose_tc()
    lat = parse_pose_latency(CMP / 'summary_13_pose_timecourse.txt')
    ax = A['C']
    if pt is not None and 'ZI' in pt['region'].values:
        s = pt[pt['region'] == 'ZI']
        for kind, col, ls, lab in (('pose', POSE, '-', 'Body pose'), ('neural', DIR, '-', 'ZI neurons'),
                                   ('resid', DIR, (0, (3, 1.5)), 'ZI, pose removed')):
            q = s[s['kind'] == kind].sort_values('t_center_ms')
            ax.plot(q['t_center_ms'], q['mean'], color=col, ls=ls, label=lab)
            sem_band(ax, q['t_center_ms'].to_numpy(), q['mean'].to_numpy(), q['sem'].to_numpy(), col, alpha=0.15)
        chance_line(ax, 0)
        zero_vline(ax)
        ax.set(xlabel='Time from movement (ms)', ylabel='Accuracy \u2212 chance')
        legend(ax, loc='upper left')
        n = lat.get('ZI', {}).get('n', '?')
        add_leg('Fig4', f'C: direction information in sliding 60-ms windows (20-ms steps) from ZI population activity, from '
                        f'tracked body pose (paws and nose; Lightning Pose or DLC), and from ZI activity after regressing out pose, '
                        f'each matched for speed and block and expressed relative to imposter-session chance '
                        f'({n} insertions; mean ± SEM).')
    else:
        note_missing(ax, 'results_comparison/pose_timecourse.csv')

    for key, reg in (('D', 'ZI'), ('E', 'VPM' if pt is not None and 'VPM' in pt['region'].values else 'MOs')):
        ax = A[key]
        if pt is None or reg not in pt['region'].values:
            note_missing(ax, f'pose time course {reg}')
            continue
        s = pt[pt['region'] == reg]
        for kind, col, lab in (('pose', POSE, 'Body pose'), ('neural', rcol(reg), f'{reg} neurons')):
            q = s[s['kind'] == kind].sort_values('t_center_ms')
            ax.plot(q['t_center_ms'], q['mean'] / q['mean'].max(), color=col, label=lab)
        chance_line(ax, 0)
        chance_line(ax, 0.5, color='#C8C8C8', lw=0.7)
        zero_vline(ax)
        ax.set(xlabel='Time from movement (ms)', ylabel='Normalized information', ylim=(-0.12, 1.12),
               yticks=[0, 0.5, 1])
        legend(ax, loc='upper left')
        L = lat.get(reg, {})
        if 'neural' in L and 'pose only' in L:
            add_leg('Fig4', f'{key}: {reg} time courses normalised to their maximum; half-maximum at {L["neural"]:.0f} ms '
                            f'(neurons) and {L["pose only"]:.0f} ms (pose).')

    ax = A['F']
    if lat and any('diff' in v for v in lat.values()):
        regs = [r for r in REG if r in lat and 'diff' in lat[r]] + [r for r in lat if r not in REG and 'diff' in lat[r]]
        txt = []
        for xi, reg in enumerate(regs):
            L = lat[reg]
            mean_ci(ax, xi, L['diff'], L['lo'], L['hi'], color=rcol(reg), ms=6.5, cap=3)
            star_above(ax, xi, L['hi'] + 3, L['p'])
            txt.append(f'{reg} {L["diff"]:+.0f} ms [{L["lo"]:+.0f}, {L["hi"]:+.0f}], {fmt_p(L["p"])}')
        chance_line(ax, 0, color=INK, ls='-', lw=0.7)
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels(regs)
        lo = min(lat[r]['lo'] for r in regs)
        hi = max(lat[r]['hi'] for r in regs)
        ax.set(ylabel='Neural \u2212 pose latency (ms)', xlim=(-0.5, len(regs) - 0.5),
               ylim=(min(lo - 8, -25), max(hi + 15, 25)))
        add_leg('Fig4', 'F: difference between the half-maximum latencies of neural and pose direction information '
                        '(positive: neurons later than the body). Paired bootstrap over insertions, 95% CI: ' + '; '.join(txt) + '.')
    else:
        note_missing(ax, 'results_comparison/summary_13_pose_timecourse.txt')
    save(fig, 'Fig4')


# ============================================================================
# Figure 5 - direction and vigour are separate; vigour information is specific to ZI
# ============================================================================
def figure5():
    CHK = parse_checks(CMP / 'summary_15_revision_checks.txt')
    fig, A = new_fig([['A', 'B', 'C'], ['D', 'E', 'F']], 4.9)
    rng = np.random.default_rng(5)
    ss = csv(PROJ / 'results' / 'subspace_summary.csv')
    pj = PROJ / 'results' / 'subspace_projections.npz'
    if pj.exists():
        z = np.load(pj)
        tcm = z['tc'] * 1000
        for key, proj, lab in (('A', z['projC'], 'Direction axis (a.u.)'), ('B', z['projS'], 'Speed axis (a.u.)')):
            ax = A[key]
            rc = majority_right_code()
            for c, col in ((1, DIR if rc == 1 else LEFT), (0, LEFT if rc == 1 else DIR)):
                for s_, ls in zip(range(3), ((0, (1, 1.2)), (0, (3, 1.5)), '-')):
                    ax.plot(tcm, proj[3 * c + s_], color=col, ls=ls, lw=1.3)
            zero_vline(ax)
            ax.set(xlabel='Time from movement (ms)', ylabel=lab)
        h = [Line2D([], [], color=DIR, label='Rightward'), Line2D([], [], color=LEFT, label='Leftward'),
             Line2D([], [], color='none', label=' '),
             Line2D([], [], color=INK, ls=(0, (1, 1.2)), lw=1.3, label='Slow'),
             Line2D([], [], color=INK, ls=(0, (3, 1.5)), lw=1.3, label='Medium'),
             Line2D([], [], color=INK, ls='-', lw=1.3, label='Fast')]
        legend(A['A'], handles=h, loc='upper left', ncol=2, columnspacing=0.8, handlelength=1.6)
        add_leg('Fig5', 'A-B: held-out condition averages (choice × speed tercile; dotted, dashed, solid = slow, medium, fast) '
                        'projected onto the direction and speed axes estimated from the other half of the trials '
                        '(pseudo-population of all ZI units, regression subspace after PCA denoising).')
    else:
        MISSING.append('results/subspace_projections.npz (rerun 05_subspace.py)')
        if ss is not None:
            ax = A['A']
            v = ss[['choice_sep_slow', 'choice_sep_mid', 'choice_sep_fast']].iloc[0].to_numpy()
            ax.bar(range(3), v, 0.6, color=DIR, lw=0)
            ax.set_xticks(range(3))
            ax.set_xticklabels(['Slow', 'Medium', 'Fast'])
            ax.set(xlabel='Speed tercile', ylabel='Direction separation (a.u.)')
            ax = A['B']
            v = ss[['choice_AB', 'speed_AB', 'choiceA_speedB', 'speedA_choiceB']].iloc[0].to_numpy()
            ax.bar(range(4), v, 0.6, color=[DIR, VIG, '#9A9A9A', '#9A9A9A'], lw=0)
            ax.set_xticks(range(4))
            ax.set_xticklabels(['Dir\nA-B', 'Spd\nA-B', 'Dir A\nSpd B', 'Spd A\nDir B'])
            ax.set(ylabel='Angle (\u00b0)', ylim=(0, 90), yticks=[0, 30, 60, 90])

    ax = A['C']
    if ss is not None:
        r = ss.iloc[0]
        if pj.exists() and 'rnd' in z.files:
            ax.hist(z['rnd'], bins=np.arange(0, 91, 3), color=NULL_F, lw=0, density=True)
            ylim = ax.get_ylim()[1]
        else:
            ylim = 1
        ln = ax.axvline(r['angle_deg'], color=INK, lw=2)
        ln._skip_check = True
        ax.errorbar([r['angle_deg']], [ylim * 1.08], xerr=[[r['angle_deg'] - r['boot_ci_low']], [r['boot_ci_high'] - r['angle_deg']]],
                    fmt='o', color=INK, ms=4.5, capsize=2.5, lw=1.4)
        t = ax.text(r['angle_deg'] - 3, ylim * 1.2, f'{r["angle_deg"]:.0f}\u00b0', ha='right', va='bottom')
        t._skip_check = True
        ax.set(xlabel='Direction\u2013speed angle (\u00b0)', ylabel='Density' if pj.exists() else '', xlim=(0, 92),
               xticks=[0, 30, 60, 90], ylim=(0, ylim * 1.42))
        if not pj.exists():
            ax.set_yticks([])
        add_leg('Fig5', f'C: angle between the direction and speed axes, {r["angle_deg"]:.1f}° (bootstrap over units 95% CI '
                        f'{r["boot_ci_low"]:.1f}-{r["boot_ci_high"]:.1f}°; {int(r["n_units"])} units, {int(r["n_insertions"])} '
                        f'insertions). Grey: random directions within the same denoised subspace (median '
                        f'{r["random_null_median"]:.1f}°). Split-half reliability: direction {r["choice_AB"]:.0f}°, speed '
                        f'{r["speed_AB"]:.0f}°; cross-axis {r["choiceA_speedB"]:.0f}° and {r["speedA_choiceB"]:.0f}°.')
    else:
        note_missing(ax, 'results/subspace_summary.csv')

    dtc = per_region('decoding_timecourse.csv')
    ax = A['D']
    if dtc:
        for reg, tc in dtc.items():
            g = tc.groupby('t_center')['r2_speed_resid']
            m, s = g.mean(), g.agg(K.sem)
            ax.plot(m.index * 1000, m, color=rcol(reg), label=reg, lw=1.9 if reg == 'ZI' else 1.3)
            sem_band(ax, m.index.to_numpy() * 1000, m.to_numpy(), s.to_numpy(), rcol(reg), alpha=0.15)
        chance_line(ax, 0)
        zero_vline(ax)
        ax.set(xlabel='Time from movement (ms)', ylabel='Speed information (R\u00b2)')
        legend(ax, loc='upper left')
        add_leg('Fig5', 'D: cross-validated R² of ridge regression predicting log peak speed after removing the choice mean '
                        '(choice-independent vigour), 100-ms windows; mean ± SEM across insertions.')
    else:
        note_missing(ax, 'decoding_timecourse.csv')

    ax = A['E']
    if dtc:
        txt = []
        vals = {}
        for xi, (reg, tc) in enumerate(dtc.items()):
            v = tc[(tc['t_center'] >= 0.10) & (tc['t_center'] <= 0.30)].groupby('pid')['r2_speed_resid'].mean()
            vals[reg] = v
            strip(ax, xi, v, color=rcol(reg), face='white', rng=rng, size=10)
            lo, hi = boot_ci(v, rng)
            mean_ci(ax, xi + 0.3, v.mean(), lo, hi, color=rcol(reg))
            p = wil(v, 0)
            star_above(ax, xi, v.max() + 0.012, p)
            txt.append(f'{reg} {v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (n = {len(v)}, Wilcoxon > 0 {fmt_p(p)})')
        chance_line(ax, 0)
        ax.set_xticks(range(len(dtc)))
        ax.set_xticklabels(list(dtc))
        lo_all = np.nanmin(np.concatenate([v.to_numpy() for v in vals.values()]))
        hi_all = np.nanmax(np.concatenate([v.to_numpy() for v in vals.values()]))
        rng_ = hi_all - lo_all
        ax.set(ylabel='Speed information (R\u00b2)', xlim=(-0.5, len(dtc) - 0.3),
               ylim=(lo_all - 0.06 * rng_, hi_all + 0.2 * rng_))
        add_leg('Fig5', 'E: mean choice-independent speed information 100-300 ms after movement onset; open circles, '
                        'insertions; filled, mean with bootstrap 95% CI: ' + '; '.join(txt) + '.')

    ax = A['F']
    gs_ = per_region('glm_summary.csv')
    if gs_:
        regs = list(gs_)
        cnt = {}
        for xi, reg in enumerate(regs):
            r = gs_[reg].set_index(['window', 'term']).loc[('move', 'speed')]
            cnt[reg] = (int(r['n_sig']), int(r['n_units']))
            ax.bar(xi, r['frac_sig'], 0.62, color=rcol(reg), lw=0)
            ax.errorbar(xi, r['frac_sig'], yerr=[[r['frac_sig'] - r['ci_low']], [r['ci_high'] - r['frac_sig']]],
                        fmt='none', ecolor=INK, lw=1.0)
        chance_line(ax, 0.05)
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels(regs)
        top = max(gs_[r].set_index(['window', 'term']).loc[('move', 'speed'), 'ci_high'] for r in regs)
        ax.set(ylabel='Speed units (fraction)', ylim=(0, top * 1.55))
        txt = []
        if 'ZI' in cnt:
            yb = top * 1.06
            for j, reg in enumerate(regs[1:], start=1):
                a, b = cnt['ZI'], cnt[reg]
                p = CHK.get(('speed', reg))
                lab = 'mouse-level cluster bootstrap'
                if p is None:
                    _, p = stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])
                    lab = 'Fisher'
                bracket(ax, 0, j, yb, p, h=top * 0.04)
                yb += top * 0.2
                txt.append(f'ZI vs {reg} {lab} {fmt_p(p)}')
        add_leg('Fig5', 'F: fraction of units with a significant speed effect in the movement window (GLM with all controls; '
                        '95% binomial CI): ' + ', '.join(f'{r} {cnt[r][0]}/{cnt[r][1]}' for r in regs) + '; ' + '; '.join(txt) + '.')
    save(fig, 'Fig5')


# ============================================================================
# Figure 6 - comparison with secondary motor cortex and sensory thalamus
# ============================================================================
def figure6():
    CHK = parse_checks(CMP / 'summary_15_revision_checks.txt')
    fig, A = new_fig([['A', 'B', 'C'], ['D', 'E', 'F']], 4.9)
    rng = np.random.default_rng(6)
    gs_ = per_region('glm_summary.csv')

    ax = A['A']
    if gs_:
        regs = list(gs_)
        cnt = {}
        for xi, reg in enumerate(regs):
            r = gs_[reg].set_index(['window', 'term']).loc[('premove', 'choice')]
            cnt[reg] = (int(r['n_sig']), int(r['n_units']))
            ax.bar(xi, r['frac_sig'], 0.62, color=rcol(reg), lw=0)
            ax.errorbar(xi, r['frac_sig'], yerr=[[r['frac_sig'] - r['ci_low']], [r['ci_high'] - r['frac_sig']]],
                        fmt='none', ecolor=INK, lw=1.0)
        chance_line(ax, 0.05)
        top = max(gs_[r].set_index(['window', 'term']).loc[('premove', 'choice'), 'ci_high'] for r in regs)
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels(regs)
        ax.set(ylabel='Direction units (fraction)', ylim=(0, top * 1.55))
        txt = []
        if 'ZI' in cnt:
            yb = top * 1.06
            for j, reg in enumerate(regs[1:], start=1):
                a, b = cnt['ZI'], cnt[reg]
                p = CHK.get(('direction', reg))
                lab = 'mouse-level cluster bootstrap'
                if p is None:
                    _, p = stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])
                    lab = 'Fisher exact'
                bracket(ax, 0, j, yb, p, h=top * 0.04)
                yb += top * 0.2
                txt.append(f'ZI vs {reg} {lab} {fmt_p(p)}')
        add_leg('Fig6', 'A: units with significant pre-movement direction coding (GLM with all controls): '
                        + ', '.join(f'{r} {cnt[r][0]}/{cnt[r][1]}' for r in regs) + '; ' + '; '.join(txt) + '.')
    else:
        note_missing(ax, 'glm_summary.csv')

    ax = A['B']
    dec = per_region('decoding_sessions.csv')
    if dec:
        vals = {}
        for reg, d in dec.items():
            null = csv(REG[reg] / 'imposter_null.csv', note=False)
            chance = null['acc_matched'].mean() if null is not None else 0.5
            vals[reg] = (d['acc_matched'] - chance).dropna()
        for xi, (reg, v) in enumerate(vals.items()):
            strip(ax, xi, v, color=rcol(reg), face='white', rng=rng, size=10)
            lo, hi = boot_ci(v, rng)
            mean_ci(ax, xi + 0.3, v.mean(), lo, hi, color=rcol(reg))
        chance_line(ax, 0)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(list(vals))
        top = max(v.max() for v in vals.values())
        ax.set(ylabel='Accuracy \u2212 chance', xlim=(-0.5, len(vals) - 0.3))
        txt = []
        if 'ZI' in vals:
            yb = top + 0.03
            for j, reg in enumerate(list(vals)[1:], start=1):
                p = stats.mannwhitneyu(vals['ZI'], vals[reg]).pvalue
                bracket(ax, 0, j, yb, p, h=0.015)
                yb += 0.07
                txt.append(f'ZI vs {reg} Mann-Whitney {fmt_p(p)}')
            ax.set_ylim(top=yb + 0.03)
        add_leg('Fig6', 'B: speed-matched pre-movement decoding minus each region\'s imposter-session chance; open circles, '
                        'insertions; filled, mean with bootstrap 95% CI. '
                        + ', '.join(f'{r} {v.mean():+.3f} (n = {len(v)})' for r, v in vals.items()) + '; ' + '; '.join(txt) + '.')
    else:
        note_missing(ax, 'decoding_sessions.csv')

    ax = A['C']
    um = csv(CMP / 'unit_matched_decoding.csv')
    if um is not None:
        Ns = np.sort(um['N'].unique())
        regs = [r for r in REG if r in um['region'].values]
        offs = np.linspace(-0.12, 0.12, len(regs)) if len(regs) > 1 else [0]
        xpos = {n: i for i, n in enumerate(Ns)}
        top_at = {}
        for off, reg in zip(offs, regs):
            g = um[um['region'] == reg]
            ms, los, his, xs = [], [], [], []
            for n in Ns:
                v = g.loc[g['N'] == n, 'above']
                if len(v) < 5:
                    continue
                lo, hi = boot_ci(v, rng)
                ms.append(v.mean())
                los.append(lo)
                his.append(hi)
                xs.append(xpos[n] + off)
                top_at[n] = max(top_at.get(n, -np.inf), hi)
            ms, los, his = map(np.array, (ms, los, his))
            ax.errorbar(xs, ms, yerr=[ms - los, his - ms], color=rcol(reg), marker='o', ms=4.5, lw=1.5, label=reg, capsize=0)
        txt = []
        if 'ZI' in regs and 'VPM' in regs:
            for n in Ns:
                z_ = um[(um['region'] == 'ZI') & (um['N'] == n)]['above']
                o = um[(um['region'] == 'VPM') & (um['N'] == n)]['above']
                if len(z_) >= 5 and len(o) >= 5:
                    p = stats.mannwhitneyu(z_, o).pvalue
                    star_above(ax, xpos[n], top_at[n] + 0.008, p)
                    txt.append(f'N = {n}: {fmt_p(p)}')
        chance_line(ax, 0)
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set(xlabel='Units per decoder', ylabel='Accuracy \u2212 chance')
        ax.set_ylim(top=ax.get_ylim()[1] + 0.03)
        legend(ax, loc='upper left')
        add_leg('Fig6', 'C: decoding with the same number of randomly drawn units in every region (20 random subsets per '
                        'insertion; mean with bootstrap 95% CI across insertions). Stars: ZI vs VPM, Mann-Whitney '
                        + '; '.join(txt) + '.')
    else:
        note_missing(ax, 'results_comparison/unit_matched_decoding.csv')

    ax = A['D']
    if dec:
        vals = {reg: (d['acc_corr_to_error'] - 0.5).dropna() for reg, d in dec.items()}
        txt = []
        for xi, (reg, v) in enumerate(vals.items()):
            strip(ax, xi, v, color=rcol(reg), face='white', rng=rng, size=10)
            lo, hi = boot_ci(v, rng)
            mean_ci(ax, xi + 0.3, v.mean(), lo, hi, color=rcol(reg))
            p = wil(v, 0)
            star_above(ax, xi, v.max() + 0.02, p)
            txt.append(f'{reg} {v.mean():+.3f} ({fmt_p(p)})')
        chance_line(ax, 0)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(list(vals))
        ax.set(ylabel='Error-trial accuracy \u2212 0.5', xlim=(-0.5, len(vals) - 0.3))
        ax.set_ylim(top=ax.get_ylim()[1] + 0.04)
        add_leg('Fig6', 'D: generalisation from correct to error trials, Wilcoxon > 0: ' + '; '.join(txt) + '.')

    ax = A['E']
    lat_u = per_region('followup_lateralisation_units.csv')
    if lat_u:
        txt = []
        for xi, (reg, lu) in enumerate(lat_u.items()):
            sig = lu[lu['p_perm'] < C.ALPHA]
            k, n = int(sig['prefers_contra'].sum()), len(sig)
            if n == 0:
                continue
            lo, hi = K.binom_ci(k, n)
            ax.bar(xi, k / n, 0.62, color=rcol(reg), lw=0)
            ax.errorbar(xi, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt='none', ecolor=INK, lw=1.0)
            p = stats.binomtest(k, n, 0.5).pvalue
            star_above(ax, xi, hi + 0.015, p)
            txt.append(f'{reg} {k}/{n} = {k / n:.1%} (binomial {fmt_p(p)})')
        chance_line(ax, 0.5)
        ax.set_xticks(range(len(lat_u)))
        ax.set_xticklabels(list(lat_u))
        ax.set(ylabel='Prefer contralateral', ylim=(0.3, 0.9), yticks=[0.3, 0.5, 0.7, 0.9])
        add_leg('Fig6', 'E: fraction of direction units whose preferred choice reports the contralateral stimulus '
                        '(95% binomial CI): ' + '; '.join(txt) + '.')
    else:
        note_missing(ax, 'followup_lateralisation_units.csv')

    ax = A['F']
    ps = per_region('pose_sessions.csv')
    if ps:
        txt = []
        for xi, (reg, d) in enumerate(ps.items()):
            for off, col, face in ((-0.17, 'above_neural', rcol(reg)), (0.17, 'above_resid', 'white')):
                v = d[col].dropna()
                lo, hi = boot_ci(v, rng)
                ax.bar(xi + off, v.mean(), 0.32, color=face, edgecolor=rcol(reg), lw=1.3)
                ax.errorbar(xi + off, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt='none', ecolor=INK, lw=1.0)
            ret = d['above_resid'].mean() / d['above_neural'].mean()
            txt.append(f'{reg} {d["above_neural"].mean():+.3f} → {d["above_resid"].mean():+.3f} '
                       f'(retention {ret:.2f}; pose alone {d["above_pose"].mean():+.3f}; n = {len(d)})')
        chance_line(ax, 0)
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels(list(ps))
        h = [Patch(facecolor='#555555', edgecolor='#555555', label='Neural'),
             Patch(facecolor='white', edgecolor='#555555', lw=1.3, label='Pose removed')]
        ax.set(ylabel='Accuracy \u2212 chance')
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]), top=ax.get_ylim()[1] * 1.3)
        legend(ax, handles=h, loc='upper right', handlelength=1.0)
        add_leg('Fig6', 'F: pre-movement direction decoding before and after regressing tracked body pose out of the neural '
                        'activity (mean, bootstrap 95% CI): ' + '; '.join(txt) + '.')
    else:
        note_missing(ax, 'pose_sessions.csv')
    save(fig, 'Fig6')


# ============================================================================
# Supplementary figures
# ============================================================================
def figureS1():
    """GLM, all windows (ZI)."""
    fig, A = new_fig([['A', 'B', 'C']], 2.9)
    summ = csv(PROJ / 'results' / 'glm_summary.csv')
    if summ is None:
        return
    terms = [t_ for t_ in ('choice', 'stimulus', 'prior', 'speed', 'rt', 'dur', 'me') if t_ in summ['term'].values]
    y = np.arange(len(terms))[::-1]
    xmax = summ['ci_high'].max() * 1.08
    for key, w, ttl in (('A', 'iti', 'ITI'), ('B', 'premove', 'Pre-movement'), ('C', 'move', 'Movement')):
        ax = A[key]
        s = summ[summ['window'] == w].set_index('term').reindex(terms)
        for yi, t_ in zip(y, terms):
            sp = ax.fill_betweenx([yi - 0.4, yi + 0.4], 0, s.loc[t_, 'null_frac_95'], color=NULL_F, lw=0)
            sp._skip_check = True
        ax.barh(y, s['frac_sig'], 0.6, color=[TERM_COL[t_] for t_ in terms], lw=0)
        ax.errorbar(s['frac_sig'], y, xerr=[s['frac_sig'] - s['ci_low'], s['ci_high'] - s['frac_sig']], fmt='none',
                    ecolor=INK, lw=0.8)
        ax.scatter(s['frac_sig_fdr'], y, marker='|', s=70, color=INK, lw=1.6, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([TERM_LAB[t_] for t_ in terms] if key == 'A' else [])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.set(xlabel='Fraction of units', xlim=(0, xmax), title=ttl)
    add_leg('FigS1', 'Single-unit GLM in all three windows (ITI, −400 to −100 ms from stimulus onset; pre-movement, −160 to 0 ms; '
                     'movement, 0-200 ms from movement onset). Bars, Freedman-Lane permutation P < 0.05 (95% binomial CI); '
                     'vertical ticks, FDR q < 0.05; grey, 95th percentile of the pooled permutation null.')
    save(fig, 'FigS1')


def figureS2():
    """Imposter-session null for all metrics (ZI)."""
    null = csv(PROJ / 'results' / 'imposter_null.csv')
    imp = parse_imposter(PROJ / 'results' / 'summary_07_imposter.txt')
    if null is None:
        return
    fig, A = new_fig([['A', 'B', 'C'], ['D', 'E', '.']], 4.3)
    items = [('premove_choice', 'Choice units (premove)', 'Direction units, pre-move', DIR, 'Fraction of units'),
             ('move_speed', 'Speed units (move)', 'Speed units, movement', VIG, 'Fraction of units'),
             ('iti_prior', 'Prior units (iti)', 'Prior units, ITI', PRIOR, 'Fraction of units'),
             ('acc_matched', 'Choice decoding (matched)', 'Decoding, speed-matched', DIR, 'Accuracy'),
             ('acc_unmatched', 'Choice decoding (unmatched)', 'Decoding, all trials', DIR, 'Accuracy')]
    txt = []
    for key, (col, name, ttl, colr, xl) in zip('ABCDE', items):
        ax = A[key]
        v = null[col].dropna().to_numpy()
        ax.hist(v, bins=20, color=NULL_F, lw=0)
        real = imp.get(name, {}).get('real', np.nan)
        if np.isfinite(real):
            ln = ax.axvline(real, color=colr, lw=2)
            ln._skip_check = True
        lo, hi = min(v.min(), real), max(v.max(), real)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel=xl, ylabel='Pairings', title=ttl, xlim=(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo)))
        p = imp.get(name, {}).get('p', (1 + (v >= real).sum()) / (1 + len(v)))
        txt.append(f'{ttl}: real {real:.3f}, imposter {v.mean():.3f}, P = {p:.3g}')
    add_leg('FigS2', 'Imposter-session null (neural data of one session paired with the behaviour of another). Grey, null '
                     'distribution; line, real value. ' + '; '.join(txt) + '.')
    save(fig, 'FigS2')


def figureS3():
    """Anatomical distribution of ZI coding types."""
    lu = csv(PROJ / 'results' / 'location_units.csv')
    if lu is None:
        return
    fig, A = new_fig([['A', 'B', 'C'], ['D', 'E', 'F']], 4.9)
    cols = {'not speed-tuned': '#CFCFCF', 'speed -': '#6B4C9A', 'speed +': VIG}
    labs = {'not speed-tuned': 'Not speed-tuned', 'speed -': 'Speed \u2212', 'speed +': 'Speed +'}
    for key, (cx, cy, xl, yl) in zip('ABC', (('y', 'z', 'AP (mm)', 'DV (mm)'), ('x', 'z', 'ML (mm)', 'DV (mm)'),
                                             ('x', 'y', 'ML (mm)', 'AP (mm)'))):
        ax = A[key]
        for grp in ('not speed-tuned', 'speed -', 'speed +'):
            s = lu[lu['speed_group'] == grp]
            sc = ax.scatter(s[cx], s[cy], s=9, color=cols[grp], lw=0, alpha=0.9, label=labs[grp])
            sc._skip_check = True
        chs = lu[lu['choice_group'] == 'choice']
        sc = ax.scatter(chs[cx], chs[cy], s=20, facecolor='none', edgecolor=DIR, lw=0.8, label='Direction unit')
        sc._skip_check = True
        ax.set(xlabel=xl, ylabel=yl)
        ax.margins(0.08)
    hs, ls_ = A['A'].get_legend_handles_labels()
    fig.legend(hs, ls_, loc='outside upper center', ncol=4, fontsize=FS['legend'], handletextpad=0.2,
               columnspacing=1.2, markerscale=1.4)
    rng = np.random.default_rng(8)
    txt = []
    for key, (col, nm) in zip('DEF', (('y', 'AP'), ('z', 'DV'), ('x', 'ML'))):
        ax = A[key]
        groups = [('speed +', 'speed_group', VIG, 'Spd +'), ('speed -', 'speed_group', '#6B4C9A', 'Spd \u2212'),
                  ('choice', 'choice_group', DIR, 'Dir'), ('non-choice', 'choice_group', '#9A9A9A', 'Other')]
        for xi, (gname, gcol, c, _) in enumerate(groups):
            v = lu.loc[lu[gcol] == gname, col].dropna()
            strip(ax, xi, v, color=c, face='white', rng=rng, size=7, alpha=0.6)
            if len(v):
                ln, = ax.plot([xi - 0.28, xi + 0.28], [v.median()] * 2, color=INK, lw=2)
                ln._skip_check = True
        a = lu.loc[lu['speed_group'] == 'speed +', col].dropna()
        b = lu.loc[lu['speed_group'] == 'speed -', col].dropna()
        p1 = stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
        a = lu.loc[lu['choice_group'] == 'choice', col].dropna()
        b = lu.loc[lu['choice_group'] == 'non-choice', col].dropna()
        p2 = stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
        ax.set_xticks(range(4))
        ax.set_xticklabels([g_[3] for g_ in groups])
        ax.set(ylabel=f'{nm} (mm)')
        txt.append(f'{nm}: speed + vs − {fmt_p(p1)}, direction vs other {fmt_p(p2)}')
    add_leg('FigS3', 'Locations of ZI units by coding type (A-C; open blue circles, pre-movement direction units) and '
                     'their AP, DV and ML coordinates (D-F; line, median). Mann-Whitney: ' + '; '.join(txt) +
                     ' (see Results for the mouse-level bootstrap).')
    save(fig, 'FigS3')


def figureS4():
    """Neural vs pose time courses for every region."""
    pt = pose_tc()
    if pt is None:
        return
    regs = [r for r in REG if r in pt['region'].values] + [r for r in pt['region'].unique() if r not in REG]
    mos = [[f'{chr(65 + 2 * i)}', f'{chr(66 + 2 * i)}'] for i in range(len(regs))]
    fig, A = new_fig(mos, 1.95 * len(regs) + 0.3, width=5.2)
    for i, reg in enumerate(regs):
        s = pt[pt['region'] == reg]
        a1, a2 = A[mos[i][0]], A[mos[i][1]]
        for kind, col, ls, lab in (('pose', POSE, '-', 'Body pose'), ('neural', rcol(reg), '-', f'{reg} neurons'),
                                   ('resid', rcol(reg), (0, (3, 1.5)), 'Pose removed')):
            q = s[s['kind'] == kind].sort_values('t_center_ms')
            a1.plot(q['t_center_ms'], q['mean'], color=col, ls=ls, label=lab)
            sem_band(a1, q['t_center_ms'].to_numpy(), q['mean'].to_numpy(), q['sem'].to_numpy(), col, alpha=0.15)
            if kind != 'resid':
                a2.plot(q['t_center_ms'], q['mean'] / q['mean'].max(), color=col, label=lab)
        for a in (a1, a2):
            chance_line(a, 0)
            zero_vline(a)
            a.set_xlabel('Time from movement (ms)' if i == len(regs) - 1 else '')
        a1.set_ylabel('Accuracy \u2212 chance')
        a2.set(ylabel='Normalized', ylim=(-0.15, 1.12), yticks=[0, 0.5, 1])
        legend(a1, loc='upper left', fontsize=7.5)
    add_leg('FigS4', 'Direction information from neurons, from body pose and from neurons with pose regressed out, for each '
                     'region (left), and the neural and pose curves normalised to their maxima (right).')
    save(fig, 'FigS4')


def figureS5():
    """Timing relative to the wheel, per region."""
    fig, A = new_fig([['A', 'B', 'C']], 2.6, width_ratios=[1.35, 1, 1])
    rc = csv(CMP / 'region_comparison.csv')
    ax = A['A']
    any_ = False
    for reg, d in REG.items():
        f = d / 'onset_sessions.npz'
        if not f.exists():
            continue
        z = np.load(f)
        diff = z['real'] - z['imp_mean']
        m, s = diff.mean(0), diff.std(0, ddof=1) / np.sqrt(len(diff))
        ax.plot(z['centers'] * 1000, m, color=rcol(reg), label=reg)
        sem_band(ax, z['centers'] * 1000, m, s, rcol(reg), alpha=0.15)
        any_ = True
    if any_:
        chance_line(ax, 0)
        zero_vline(ax)
        ax.set(xlabel='Time from movement (ms)', ylabel='Accuracy \u2212 chance')
        legend(ax, loc='upper left')
    if rc is not None:
        for key, col, lo, hi, yl in (('B', 'latency', 'lat_lo', 'lat_hi', 'Half-max latency (ms)'),
                                     ('C', 'reversal_cross', 'rev_lo', 'rev_hi', 'Switch time (ms)')):
            ax = A[key]
            r = rc.set_index('region')
            regs = [g for g in REG if g in r.index and col in r and np.isfinite(r.loc[g, col])]
            for xi, reg in enumerate(regs):
                mean_ci(ax, xi, r.loc[reg, col], r.loc[reg, lo], r.loc[reg, hi], color=rcol(reg), ms=6, cap=3)
            chance_line(ax, 0, color=INK, ls='-', lw=0.7)
            ax.set_xticks(range(len(regs)))
            ax.set_xticklabels(regs)
            ax.set(ylabel=yl, xlim=(-0.5, len(regs) - 0.5))
        add_leg('FigS5', 'A: time-resolved direction decoding minus imposter chance (speed, block and previous choice matched). '
                         'B: half-maximum latency relative to detected wheel movement onset (bootstrap 95% CI). C: time at which '
                         'the decoder switches to the final choice relative to the wheel reversal on change-of-mind trials. '
                         'Both are measured relative to the wheel; tracked body movement begins at a similar time (Fig. 4).')
    save(fig, 'FigS5')


# ----------------------------------------------------------------------------
FIGS = {'1': figure1, '2': figure2, '3': figure3, '4': figure4, '5': figure5, '6': figure6,
        'S1': figureS1, 'S2': figureS2, 'S3': figureS3, 'S4': figureS4, 'S5': figureS5}


def write_legends():
    L = ['# Figure legends - numbers for the text', '',
         'Generated by 14_make_figures.py from the saved results.', '']
    for k, v in LEGENDS.items():
        L += [f'## {k}', ''] + [f'- {x}' for x in v] + ['']
    if WARN:
        L += ['## Layout warnings', ''] + [f'- {w}' for w in WARN] + ['']
    (OUT / 'figure_legends.md').write_text('\n'.join(L), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', help='figure ids, e.g. 1 4 S2')
    args = ap.parse_args()
    setup_style()
    REG.update(region_dirs())
    print('regions found:', ', '.join(REG) or 'none')
    OUT.mkdir(exist_ok=True)
    for k in (args.only or FIGS):
        print(f'Figure {k}')
        try:
            FIGS[k]()
        except Exception as e:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'  Figure {k} failed: {e}')
        plt.close('all')
    write_legends()
    if MISSING:
        print('\nMissing inputs (panels left empty):')
        for m in dict.fromkeys(MISSING):
            print('  -', m)
    print(f'\n{len(WARN)} layout warning(s). Figures in {OUT}')


if __name__ == '__main__':
    main()
