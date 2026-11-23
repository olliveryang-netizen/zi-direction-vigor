"""
Step 06 - Anatomical organisation of coding types within ZI (paper Figure S3).

Uses the unit-level GLM results (step 03) and CCF coordinates of each unit.
  * speed-tuned units split by sign: positive (fire more on faster movements) vs negative
  * choice-selective units (premove window) vs non-selective
Compared along AP / DV / ML (Mann-Whitney U), with a mouse-level cluster bootstrap of the
difference in mean position (resampling mice, so units from the same animal are not treated
as independent).

Outputs: results/location_units.csv, results/summary_06_location.txt, figures/step06_location.(png|pdf)
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C

N_BOOT = 2000


def cluster_boot_diff(df, col, grp_a, grp_b, rng):
    """Difference in mean coordinate (a - b), bootstrap resampling mice."""
    mice = df['subject'].unique()
    diffs = []
    for _ in range(N_BOOT):
        s = pd.concat([df[df['subject'] == m] for m in rng.choice(mice, len(mice))])
        a, b = s.loc[s['group'] == grp_a, col], s.loc[s['group'] == grp_b, col]
        if len(a) > 1 and len(b) > 1:
            diffs.append(a.mean() - b.mean())
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return np.percentile(diffs, [2.5, 97.5]), max(p, 1 / len(diffs))


def main():
    rng = np.random.default_rng(C.RANDOM_SEED + 6)
    plt = K.setup_matplotlib()
    g = pd.read_csv(C.RESULTS_DIR / 'glm_units.csv')
    key = ['pid', 'unit_index']
    base = g.drop_duplicates(key)[key + ['subject', 'x', 'y', 'z']].copy()
    sp = g[(g['window'] == 'move') & (g['term'] == 'speed')].set_index(key)
    ch = g[(g['window'] == 'premove') & (g['term'] == 'choice')].set_index(key)
    base = base.set_index(key)
    base['speed_sig'] = sp['p_perm'] < C.ALPHA
    base['speed_beta'] = sp['beta']
    base['choice_sig'] = ch['p_perm'] < C.ALPHA
    base = base.reset_index()
    for c in ('x', 'y', 'z'):
        base[c] = base[c] * 1e3          # metres -> mm
    base['speed_group'] = np.where(~base['speed_sig'], 'not speed-tuned',
                                   np.where(base['speed_beta'] > 0, 'speed +', 'speed -'))
    base['choice_group'] = np.where(base['choice_sig'], 'choice', 'non-choice')
    base.to_csv(C.RESULTS_DIR / 'location_units.csv', index=False)

    lines = ['=== Anatomical organisation (coordinates in mm from bregma; IBL convention x=ML, y=AP, z=DV) ===']
    tests = {}
    for label, gcol, a, b in (('speed + vs speed -', 'speed_group', 'speed +', 'speed -'),
                              ('choice vs non-choice', 'choice_group', 'choice', 'non-choice')):
        d = base.assign(group=base[gcol])
        lines.append(f'--- {label}: n = {(d.group == a).sum()} vs {(d.group == b).sum()}')
        for col, nm in (('y', 'AP'), ('z', 'DV'), ('x', 'ML')):
            xa, xb = d.loc[d.group == a, col].dropna(), d.loc[d.group == b, col].dropna()
            if len(xa) < 3 or len(xb) < 3:
                continue
            pu = stats.mannwhitneyu(xa, xb).pvalue
            ci, pb = cluster_boot_diff(d, col, a, b, rng)
            tests[(label, nm)] = (xa.mean() - xb.mean(), pu, pb)
            lines.append(f'{nm}: mean diff {xa.mean() - xb.mean():+.3f} mm, Mann-Whitney p={pu:.3g}; '
                         f'mouse-bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], p={pb:.3g}')
    if (base['speed_group'] != 'not speed-tuned').any():
        st = base[base['speed_group'] != 'not speed-tuned']
        lines.append(f"speed-tuned units: {len(st)}; positive {(st.speed_group == 'speed +').mean():.1%}, "
                     f"negative {(st.speed_group == 'speed -').mean():.1%}")
    C.write_summary('summary_06_location', lines)

    # ---------------- figure ----------------
    cols = {'speed +': K.COL_SPEED, 'speed -': '#5a3e8a', 'not speed-tuned': '0.8'}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8))
    fig.subplots_adjust(hspace=0.6, wspace=0.55)
    for ax, (cx, cy, xl, yl, ttl) in zip(axs[0], (('y', 'z', 'AP (mm)', 'DV (mm)', 'A  Sagittal'),
                                                  ('x', 'z', 'ML (mm)', 'DV (mm)', 'B  Coronal'),
                                                  ('x', 'y', 'ML (mm)', 'AP (mm)', 'C  Horizontal'))):
        for grp in ('not speed-tuned', 'speed -', 'speed +'):
            s = base[base['speed_group'] == grp]
            ax.scatter(s[cx], s[cy], s=6, color=cols[grp], lw=0, alpha=0.8, label=grp)
        ax.set(xlabel=xl, ylabel=yl, title=ttl)
    axs[0, 0].legend(frameon=True, framealpha=0.85, edgecolor='none', fontsize=5.5, markerscale=1.5, loc='lower left')
    for ax, (col, nm) in zip(axs[1], (('y', 'AP'), ('z', 'DV'), ('x', 'ML'))):
        data = [base.loc[base['speed_group'] == gname, col].dropna() for gname in ('speed +', 'speed -')]
        data += [base.loc[base['choice_group'] == gname, col].dropna() for gname in ('choice', 'non-choice')]
        bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True)
        for patch, c in zip(bp['boxes'], [K.COL_SPEED, '#5a3e8a', K.COL_CHOICE, '0.8']):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(['spd+', 'spd-', 'choice', 'non-\nchoice'], fontsize=7)
        t1 = tests.get(('speed + vs speed -', nm))
        t2 = tests.get(('choice vs non-choice', nm))
        ttl = f"{'DEF'[['AP', 'DV', 'ML'].index(nm)]}  {nm}"
        if t1:
            ttl += f'\nspd p={t1[2]:.2g}'
        if t2:
            ttl += f', choice p={t2[2]:.2g}'
        ax.set(ylabel=f'{nm} (mm)', title=ttl)
    C.save_fig(fig, 'step06_location')


if __name__ == '__main__':
    main()
