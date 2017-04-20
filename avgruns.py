import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######################### Crunch data ##########################

dfs = pd.DataFrame()
for i in range(1,9):
    dfs = dfs.append(pd.DataFrame.from_csv("/tmp/t/lcsratio_trainsize_beta_mu_{}.log".format(i), sep='\t', header=0, index_col=['trainsize','feature_structures','oov_handling','mu','beta']).applymap(lambda x: round(x, 7) if isinstance(x, float) else x))

df = dfs.xs('lcsratio', level='feature_structures')

######################### Plot data ##########################

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif=['charter'])

fig, ax = plt.subplots(3, 3, figsize=(10, 10)) #, sharex=True, sharey=True)

tss = [10,50,100,500,1000,5000,10000,20000,39000]

for i in range(3):
    for j in range(3):
        ts = tss[i + 3*j]
        
        if (df.index.get_level_values('trainsize') == ts).any():
            tsdf = df.xs(ts, level='trainsize').reset_index()
            means = tsdf.pivot_table(index='mu', columns='beta', values='fmeasure', aggfunc=np.mean)
            errs = tsdf.pivot_table(index='mu', columns='beta', values='fmeasure', aggfunc=np.std)
            means.plot(ax=ax[j][i], marker='o', markersize=2, yerr=errs)

        ax[j][i].set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
        ax[j][i].set_xlabel("$\\mu$", labelpad=0)
        ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
        ax[j][i].legend(loc='lower right', title='$\\beta$', prop={'size':7})
        ax[j][i].grid(True)

fig.tight_layout()
fig.savefig('/tmp/test.pdf', format='pdf', dpi=1000)
