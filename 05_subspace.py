"""
Step 05 - Population subspace analysis (paper Figure 5A-C).

Are choice and movement-vigour signals carried along SEPARATE population dimensions in ZI?

Pseudo-population of all ZI units (sessions combined). For every unit, sqrt spike counts aligned to
movement onset (-300..+300 ms, 20-ms bins, smoothed) are z-scored, and at each time bin regressed on
  choice, log peak speed, stimulus side, |contrast|, block prior, trial drift   (Mante et al., 2013 style)
The regression coefficients across units define a CHOICE axis and a SPEED axis (time of maximal norm,
choice in -200..0 ms, speed in 0..200 ms), after denoising with the top principal components of the
condition-averaged responses (choice x speed tercile).

  * angle between axes: observed, bootstrap CI over units, null of random directions in the same subspace
  * split-half reliability: axes estimated on half A vs half B of the trials
  * cross-validated projections: axes from half A, condition averages of held-out half B projected onto them
  * invariance: choice separation on the choice axis within each speed tercile

Outputs: results/subspace_summary.csv, results/subspace_projections.npz, results/summary_05_subspace.txt,
         figures/step05_subspace.(png|pdf)
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import common as K
import config as C

T_RANGE = (-0.30, 0.30)
CHOICE_T = (-0.20, 0.00)
SPEED_T = (0.00, 0.20)
N_BOOT = 1000
VAR_KEEP = 0.90
MIN_PER_COND = 4               # trials per (choice x speed tercile) per half


def regressors(t):
    z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0
    ls = np.log(t['peak_speed'].to_numpy())
    return np.column_stack([np.ones(len(t)), t['choice'].to_numpy(float), z(ls),
                            np.sign(t['signed_contrast'].to_numpy()), z(np.abs(t['signed_contrast'].to_numpy())),
                            t['probabilityLeft'].to_numpy() - 0.5, z(t['trial_number'].to_numpy(float))])


def session_blocks(t, b, rng):
    """Return per-half betas (units x 2 x T for choice/speed) and condition averages (units x 6 x T)."""
    ok = K.valid_trials(t)
    t = t[ok].reset_index(drop=True)
    edges, cent = K.bin_centers()
    tsel = (cent >= T_RANGE[0]) & (cent <= T_RANGE[1])
    Y = np.sqrt(b['fm_counts'][:, ok][:, :, tsel].astype(float))
    Y = gaussian_filter1d(Y, 1.0, axis=-1)
    mu = Y.mean((1, 2), keepdims=True)
    sd = Y.std((1, 2), keepdims=True)
    sd[sd == 0] = 1
    Y = (Y - mu) / sd                                         # units x trials x T
    terc = pd.qcut(t['peak_speed'], 3, labels=False).to_numpy()
    cond = (t['choice'].to_numpy() == 1).astype(int) * 3 + terc   # 0..5
    # stratified random split into halves
    half = np.zeros(len(t), int)
    for c in range(6):
        i = rng.permutation(np.where(cond == c)[0])
        half[i[: len(i) // 2]] = 1
    counts = np.array([[((cond == c) & (half == h)).sum() for c in range(6)] for h in (0, 1)])
    if counts.min() < MIN_PER_COND:
        return None
    X = regressors(t)
    out = {}
    for h in ('all', 0, 1):
        sel = np.ones(len(t), bool) if h == 'all' else (half == h)
        Xs = X[sel]
        nU, _, nT = Y.shape
        Ys = Y[:, sel, :].transpose(1, 0, 2).reshape(sel.sum(), nU * nT)
        beta, *_ = np.linalg.lstsq(Xs, Ys, rcond=None)
        beta = beta.reshape(X.shape[1], nU, nT)
        ca = np.stack([Y[:, sel & (cond == c), :].mean(1) for c in range(6)], 1)   # units x 6 x T
        out[h] = dict(b_choice=beta[1], b_speed=beta[2], cond=ca)
    return out, cent[tsel]


def denoiser(cond):
    """Projection onto top PCs explaining VAR_KEEP of condition-averaged variance (units x units)."""
    M = cond.reshape(cond.shape[0], -1)
    M = M - M.mean(1, keepdims=True)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    ev = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(ev, VAR_KEEP) + 1)
    return U[:, :k] @ U[:, :k].T, k


def axis_from(beta, D, tc, rng_t):
    sel = (tc >= rng_t[0]) & (tc <= rng_t[1])
    Bd = D @ beta
    norms = np.linalg.norm(Bd[:, sel], axis=0)
    v = Bd[:, sel][:, np.argmax(norms)]
    return v / np.linalg.norm(v)


def angle(a, b):
    return np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))   # sign-invariant, 0..90 deg


def main():
    rng = np.random.default_rng(C.RANDOM_SEED + 5)
    plt = K.setup_matplotlib()
    parts, tc = [], None
    for path in K.list_insertions():
        meta, t, u, b = K.load_insertion(path)
        r = session_blocks(t, b, rng)
        if r is None:
            continue
        parts.append(r[0])
        tc = r[1]
    print(f'pseudo-population from {len(parts)} insertions')

    def cat(h, key):
        return np.concatenate([p[h][key] for p in parts], 0)

    data = {h: {k: cat(h, k) for k in ('b_choice', 'b_speed', 'cond')} for h in ('all', 0, 1)}
    nU = data['all']['cond'].shape[0]

    D, k = denoiser(data['all']['cond'])
    ax_c = axis_from(data['all']['b_choice'], D, tc, CHOICE_T)
    ax_s = axis_from(data['all']['b_speed'], D, tc, SPEED_T)
    ang = angle(ax_c, ax_s)

    # bootstrap over units
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, nU, nU)
        Db, _ = denoiser(data['all']['cond'][i])
        boot.append(angle(axis_from(data['all']['b_choice'][i], Db, tc, CHOICE_T),
                          axis_from(data['all']['b_speed'][i], Db, tc, SPEED_T)))
    boot = np.array(boot)
    # random-direction null inside the same k-dim denoised subspace
    U = np.linalg.svd(D)[0][:, :k]
    rnd = np.array([angle(*(U @ v / np.linalg.norm(U @ v) for v in rng.standard_normal((2, k))))
                    for _ in range(5000)])

    # split-half reliability and cross-validated projections
    DA, _ = denoiser(data[0]['cond'])
    DB, _ = denoiser(data[1]['cond'])
    cA, sA = axis_from(data[0]['b_choice'], DA, tc, CHOICE_T), axis_from(data[0]['b_speed'], DA, tc, SPEED_T)
    cB, sB = axis_from(data[1]['b_choice'], DB, tc, CHOICE_T), axis_from(data[1]['b_speed'], DB, tc, SPEED_T)
    rel = dict(choice_AB=angle(cA, cB), speed_AB=angle(sA, sB), choiceA_speedB=angle(cA, sB), speedA_choiceB=angle(sA, cB))
    sgn = np.sign(cA @ data[1]['b_choice'][:, (tc >= CHOICE_T[0]) & (tc <= CHOICE_T[1])].mean(1)) or 1
    sgs = np.sign(sA @ data[1]['b_speed'][:, (tc >= SPEED_T[0]) & (tc <= SPEED_T[1])].mean(1)) or 1
    projC = np.einsum('u,uct->ct', cA * sgn, data[1]['cond'])    # 6 x T (held-out half)
    projS = np.einsum('u,uct->ct', sA * sgs, data[1]['cond'])
    wc = (tc >= CHOICE_T[0]) & (tc <= CHOICE_T[1])
    ws = (tc >= SPEED_T[0]) & (tc <= SPEED_T[1])
    choice_sep = [projC[3 + s, wc].mean() - projC[s, wc].mean() for s in range(3)]    # per speed tercile
    speed_sep = [projS[3 * c + 2, ws].mean() - projS[3 * c, ws].mean() for c in range(2)]  # fast - slow per choice
    choice_on_speed = [projS[3 + s, ws].mean() - projS[s, ws].mean() for s in range(3)]

    summ = dict(n_units=nU, n_insertions=len(parts), n_pcs=k, angle_deg=ang, boot_ci_low=np.percentile(boot, 2.5),
                boot_ci_high=np.percentile(boot, 97.5), random_null_median=np.median(rnd),
                random_null_5pct=np.percentile(rnd, 5), p_more_aligned_than_random=(rnd <= ang).mean(), **rel,
                choice_sep_slow=choice_sep[0], choice_sep_mid=choice_sep[1], choice_sep_fast=choice_sep[2],
                speed_sep_choice_neg=speed_sep[0], speed_sep_choice_pos=speed_sep[1])
    pd.DataFrame([summ]).to_csv(C.RESULTS_DIR / 'subspace_summary.csv', index=False)
    # for the paper figure (14_make_figures.py): held-out projections, bootstrap and random-null angles
    np.savez(C.RESULTS_DIR / 'subspace_projections.npz', tc=tc, projC=projC, projS=projS, boot=boot, rnd=rnd)

    # ---------------- figure ----------------
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8))
    fig.subplots_adjust(hspace=0.65, wspace=0.55)
    styles = [':', '--', '-']
    for ax, proj, title, lab in ((axs[0, 0], projC, 'A  Held-out data on choice axis', 'choice axis (a.u.)'),
                                 (axs[0, 1], projS, 'B  Held-out data on speed axis', 'speed axis (a.u.)')):
        for c, col in ((1, K.COL_CHOICE), (0, '#d1495b')):
            for s in range(3):
                ax.plot(tc * 1000, proj[3 * c + s], color=col, ls=styles[s], lw=1)
        ax.axvline(0, color='k', lw=0.6)
        ax.set(xlabel='Time from movement onset (ms)', ylabel=lab, title=title)
    ax = axs[0, 2]
    for c, col in ((1, K.COL_CHOICE), (0, '#d1495b')):
        for s in range(3):
            ax.plot(projC[3 * c + s], projS[3 * c + s], color=col, ls=styles[s], lw=1)
            ax.plot(projC[3 * c + s][-1], projS[3 * c + s][-1], 'o', color=col, ms=2)
    ax.set(xlabel='choice axis', ylabel='speed axis', title='C  State space (held-out)')

    ax = axs[1, 0]
    ax.hist(rnd, bins=40, color='0.8', density=True, label='random directions')
    ax.axvline(ang, color='k', lw=1.5, label=f'observed {ang:.0f}°')
    ax.axvspan(*np.percentile(boot, [2.5, 97.5]), color=K.COL_CHOICE, alpha=0.25, label='bootstrap 95% CI')
    ax.set(xlabel='Angle choice vs speed axis (deg)', ylabel='Density', title='D  Axis alignment', xlim=(0, 90))
    ax.legend(frameon=False, fontsize=6, loc='upper left')

    ax = axs[1, 1]
    vals = [rel['choice_AB'], rel['speed_AB'], rel['choiceA_speedB'], rel['speedA_choiceB']]
    ax.bar(range(4), vals, color=[K.COL_CHOICE, K.COL_SPEED, '0.6', '0.6'])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['choice\nA vs B', 'speed\nA vs B', 'choice A\nspeed B', 'speed A\nchoice B'], fontsize=6, rotation=30, ha='right')
    ax.set(ylabel='Angle (deg)', ylim=(0, 90), title='E  Split-half reliability')

    ax = axs[1, 2]
    ax.bar(range(3), choice_sep, color=K.COL_CHOICE, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['slow', 'medium', 'fast'])
    ax.set(xlabel='Speed tercile', ylabel='Choice separation\non choice axis', title='F  Invariance to speed')
    C.save_fig(fig, 'step05_subspace')

    C.write_summary('summary_05_subspace', [
        '=== Population subspace (pseudo-population) ===',
        f'units: {nU} from {len(parts)} insertions; denoising PCs: {k} ({VAR_KEEP:.0%} variance)',
        f'angle choice vs speed axis: {ang:.1f} deg (bootstrap 95% CI {summ["boot_ci_low"]:.1f}-{summ["boot_ci_high"]:.1f}); '
        f'random-direction null median {np.median(rnd):.1f} deg, 5th pct {np.percentile(rnd, 5):.1f} deg; '
        f'P(random <= observed) = {summ["p_more_aligned_than_random"]:.3f}',
        f'split-half angles: choice A-B {rel["choice_AB"]:.1f}, speed A-B {rel["speed_AB"]:.1f}, '
        f'choice A-speed B {rel["choiceA_speedB"]:.1f}, speed A-choice B {rel["speedA_choiceB"]:.1f} deg',
        f'held-out choice separation on choice axis by speed tercile (slow/mid/fast): '
        + ' / '.join(f'{x:.2f}' for x in choice_sep),
        f'held-out speed separation (fast-slow) on speed axis, choice -1 / +1: ' + ' / '.join(f'{x:.2f}' for x in speed_sep),
        f'choice separation on SPEED axis by tercile: ' + ' / '.join(f'{x:.2f}' for x in choice_on_speed),
        'Note: pseudo-population ignores noise correlations between simultaneously recorded units.',
    ])


if __name__ == '__main__':
    main()
