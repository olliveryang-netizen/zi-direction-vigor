"""
Step 02 - Dataset and behaviour (paper Figure 1).

  1A  Locations of ZI units (sagittal and coronal projections, CCF coordinates relative to bregma)
  1B  Psychometric curves by block prior (mean +/- SEM across sessions)
  1C  Reaction-time distribution
  1D  The choice-movement confound: per-session difference in peak wheel speed between choices
  1E  Units per insertion

Outputs: figures/step02_dataset_behavior.(png|pdf), results/table1_dataset.csv, results/summary_02_dataset.txt
"""
import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C


def main():
    plt = K.setup_matplotlib()
    rows, units_all, psy, rts, conf = [], [], [], [], []
    for path in K.list_insertions():
        meta, t, u, _ = K.load_insertion(path)
        rows.append(dict(pid=meta['pid'], eid=meta['eid'], subject=meta['subject'], lab=meta['lab'],
                         n_units=len(u), n_trials=len(t), pct_correct=t['correct'].mean() * 100,
                         n_error=int((t['correct'] == 0).sum()), n_zero_contrast=int((t['signed_contrast'] == 0).sum()),
                         median_rt_ms=np.median(t['rt']) * 1000,
                         has_motion_energy=bool(meta.get('motion_energy'))))
        units_all.append(u)
        for blk, g in t.groupby('probabilityLeft'):
            for c, gg in g.groupby('signed_contrast'):
                psy.append(dict(pid=meta['pid'], block=blk, contrast=c, p_choice_pos=(gg['choice'] == 1).mean(), n=len(gg)))
        rts.append(t['rt'].to_numpy())
        ok = K.valid_trials(t)
        a = t.loc[ok & (t['choice'] == 1), 'peak_speed']
        b = t.loc[ok & (t['choice'] == -1), 'peak_speed']
        d = (np.log(a).mean() - np.log(b).mean()) / np.sqrt((np.log(a).var() + np.log(b).var()) / 2)
        conf.append(dict(pid=meta['pid'], subject=meta['subject'], cohen_d_logspeed=d,
                         p=stats.mannwhitneyu(a, b).pvalue))

    tab = pd.DataFrame(rows)
    tab.to_csv(C.RESULTS_DIR / 'table1_dataset.csv', index=False)
    units = pd.concat(units_all, ignore_index=True)
    psy = pd.DataFrame(psy)
    conf = pd.DataFrame(conf)
    conf.to_csv(C.RESULTS_DIR / 'choice_speed_confound.csv', index=False)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.85)

    # 1A unit locations (IBL coordinates in metres -> mm)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(units['y'] * 1e3, units['z'] * 1e3, s=4, c=K.COL_GREY, alpha=0.6, lw=0)
    ax.set(xlabel='AP (mm from bregma)', ylabel='DV (mm)', title=f'A  {C.REGION} units (sagittal)')
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(units['x'] * 1e3, units['z'] * 1e3, s=4, c=K.COL_GREY, alpha=0.6, lw=0)
    ax.set(xlabel='ML (mm)', ylabel='DV (mm)', title='coronal')

    # 1B psychometric curves
    ax = fig.add_subplot(gs[0, 2:])
    cols = {0.2: '#d1495b', 0.5: '0.3', 0.8: '#00798c'}
    for blk, g in psy.groupby('block'):
        m = g.groupby('contrast')['p_choice_pos'].agg(['mean', K.sem])
        ax.errorbar(m.index * 100, m['mean'], yerr=m['sem'], color=cols.get(blk, 'k'), marker='o', ms=3,
                    lw=1, capsize=0, label=f'P(left stim) = {blk}')
    ax.set(xlabel='Signed contrast (%)  (right - left)', ylabel='P(choice = +1)', ylim=(0, 1),
           title='B  Behaviour in ZI sessions')
    ax.legend(frameon=False, fontsize=6)

    # 1C RT
    ax = fig.add_subplot(gs[1, 0])
    allrt = np.concatenate(rts) * 1000
    ax.hist(allrt[allrt < 1000], bins=40, color=K.COL_GREY)
    ax.set(xlabel='Reaction time (ms)', ylabel='Trials', title='C  Reaction time')

    # 1D confound
    ax = fig.add_subplot(gs[1, 1:3])
    cs = conf.sort_values('cohen_d_logspeed')
    colr = np.where(cs['p'] < 0.05, K.COL_SPEED, '0.75')
    ax.bar(np.arange(len(cs)), cs['cohen_d_logspeed'], color=colr, width=0.8)
    ax.axhline(0, color='k', lw=0.6)
    ax.set(xlabel='Insertion (sorted)', ylabel="Cohen's d, log peak speed\n(choice +1 vs -1)",
           title='D  Choice and wheel speed are confounded')

    # 1E units per insertion
    ax = fig.add_subplot(gs[1, 3])
    ax.hist(tab['n_units'], bins=np.arange(0, tab['n_units'].max() + 2), color=K.COL_GREY)
    ax.set(xlabel=f'{C.REGION} units / insertion', ylabel='Insertions', title='E  Sampling')
    C.save_fig(fig, 'step02_dataset_behavior')

    n_sig_conf = int((conf['p'] < 0.05).sum())
    C.write_summary('summary_02_dataset', [
        f'=== Dataset ({C.REGION}) ===',
        f"insertions: {len(tab)} | sessions: {tab['eid'].nunique()} | mice: {tab['subject'].nunique()} | labs: {tab['lab'].nunique()}",
        f"{C.REGION} units: {tab['n_units'].sum()} (median {tab['n_units'].median():.0f}/insertion, range {tab['n_units'].min()}-{tab['n_units'].max()})",
        f"insertions with >= {C.MIN_UNITS_POPULATION} units: {(tab['n_units'] >= C.MIN_UNITS_POPULATION).sum()}",
        f"trials: {tab['n_trials'].sum()} | error trials: {tab['n_error'].sum()} | 0% contrast trials: {tab['n_zero_contrast'].sum()}",
        f"performance: {tab['pct_correct'].mean():.1f} +/- {tab['pct_correct'].std():.1f} % correct",
        f"median RT: {np.median(allrt):.0f} ms",
        f"sessions with motion energy: {tab['has_motion_energy'].sum()}",
        '=== Choice-speed confound ===',
        f"insertions where peak speed differs between choices (p<0.05): {n_sig_conf}/{len(conf)}",
        f"|Cohen's d| median: {conf['cohen_d_logspeed'].abs().median():.2f}",
    ])


if __name__ == '__main__':
    main()
