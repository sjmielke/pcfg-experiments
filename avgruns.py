import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif=['charter'])

logroot = "/home/sjm/documents/Uni/FuzzySP/pcfg-experiments/logs/"

def join_file_frames(filenames, indices):
    df = pd.DataFrame()
    for filename in filenames:
        df = df.append(pd.DataFrame.from_csv(filename, sep='\t', header=0, index_col=indices).applymap(lambda x: round(x, 7) if isinstance(x, float) else x))

    return df

def pos_gold_plots(relative):
    indices = ['trainsize','feature_structures','oov_handling','mu']
    
    pg_df = join_file_frames([logroot + "/pos_soft_matching_trainsize_mu_{}.log".format(i) for i in range(1,5)], indices)
    
    pg_df = pg_df.append(join_file_frames([logroot + "/pos_soft_matching_trainsize_mu_others.log"], indices))
    
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    pg_df = pg_df.xs('postagsonly', level='feature_structures')
    pg_df = pg_df.reset_index()
    rootdata = pg_df[pg_df.trainsize > 10]

    windowsize = 5

    facets = [("full raw data", 1, 0), ("full data, avg. window {}".format(windowsize), windowsize, 0), ("4 bins, avg. window {}".format(windowsize), windowsize, 4), ("2 bins, moving avg. window {}".format(windowsize), windowsize, 2)]

    for j in [0,1]:
        for i in [0,1]:
            (title, smoothify, bins) = facets[i + 2*j]

            pg_df = rootdata.copy()

            if bins == 4:
                pg_df['trainsize'] = pd.cut(pg_df['trainsize'], [0, 234, 1234, 12345, 45678], labels=["50/100", "500/1000", "5000/10000", "20000/39000"])
            elif bins == 2:
                pg_df['trainsize'] = pd.cut(pg_df['trainsize'], [0, 1234, 45678], labels=["50-1000", "5000-39000"])
            elif bins == 0:
                pass
            else:
                raise Exception("Wrong bins number")

            means = pg_df.pivot_table(index='mu', columns='trainsize', values='fmeasure', aggfunc=np.mean)
            errs = pg_df.pivot_table(index='mu', columns='trainsize', values='fmeasure', aggfunc=np.std)
            
            means = means.rolling(window=smoothify, min_periods=1).mean()
            
            if relative:
                means = means.apply(lambda ser: ser / ser[0])
                #means = means.apply(lambda ser: (ser - ser[100]) / (ser[0] - ser[100]))
            
            errs = means.rolling(window=smoothify, min_periods=1).std()
            
            means.plot(ax=axes[j][i], marker='o', markersize=2, yerr=errs, elinewidth=0.75)
            # lo = means - errs
            # hi = means + errs
            # try:
            #     axes[j][i].fill_between(means.index, lo['50-1000'], hi['50-1000'], alpha=0.8)
            #     axes[j][i].fill_between(means.index, lo['5000-39000'], hi['5000-39000'], alpha=0.8)
            # except:
            #     pass
            
            axes[j][i].legend(loc='upper right', title='train size', prop={'size':7})

            axes[j][i].set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
            axes[j][i].set_xlabel("$\\mu$", labelpad=0)
            axes[j][i].set(ylabel="$F_1$ measure" + " (relative from $\mu$=0)" if relative else "", title="Choice of $\\mu$ \\small ({})".format(title))
            axes[j][i].grid(True)

    fig.tight_layout()
    fig.savefig('/tmp/pos_plot.pdf', format='pdf', dpi=1000)
    fig.savefig('/tmp/pos_plot.png', format='png', dpi=1000)


def lcs_plots():
    indices = ['trainsize','feature_structures','oov_handling','mu','beta']

    lcs_df = join_file_frames([logroot + "/lcsratio_trainsize_beta_mu_{}.log".format(i) for i in range(1,9)] + [logroot + "/lcsratio_beta_50.log", logroot + "/lcsratio_mu_500_1000.log"], indices)

    lcs_df_2 = lcs_df.append(join_file_frames([logroot + "/lcsratio_beta_20_mu_100.log", logroot + "/lcsratio_beta_6_8_12_14_17_20_mu_100.log", logroot + "/lcsratio_beta_60_70_80_90_100_mu_0_100.log", logroot + "/lcsratio_beta_6_8_12_14_17_20_mu_0.log", logroot + "/lcsratio_beta_30_40_mu_0_100.log"], indices))

    tss = [10,50,100,500,1000,5000,10000,20000,39000]

    def lcs_ratio_monsterplot():
        # tells us that mu = 100 is a good choice

        fig, ax = plt.subplots(3, 3, figsize=(10, 10)) #, sharex=True, sharey=True)

        for j in range(3):
            for i in range(3):
                ts = tss[i + 3*j]
                
                if (lcs_df.index.get_level_values('trainsize') == ts).any():
                    tsdf = lcs_df.xs(ts, level='trainsize')
                    lcs_only_df = tsdf.xs('lcsratio', level='feature_structures').reset_index()
                    
                    means = lcs_only_df.pivot_table(index='mu', columns='beta', values='fmeasure', aggfunc=np.mean)
                    errs = lcs_only_df.pivot_table(index='mu', columns='beta', values='fmeasure', aggfunc=np.std)
                    means.plot(ax=ax[j][i], marker='o', markersize=2, yerr=errs)
                    
                    baseline = lcs_df.loc[ts].loc["exactmatch"].loc["uniform"].loc[0.0].loc[1.0]
                    baseline = baseline['fmeasure']
                    baseline = np.mean(baseline)
                    
                    ax[j][i].axhline(baseline, 0.0, 1.0, linestyle='dashed', linewidth=1, color='black')

                ax[j][i].set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
                ax[j][i].set_xlabel("$\\mu$", labelpad=0)
                ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
                ax[j][i].legend(loc='lower right', title='$\\beta$', prop={'size':7})
                ax[j][i].grid(True)

        fig.tight_layout()
        fig.savefig('/tmp/lcs_ratio_monsterplot.pdf', format='pdf', dpi=1000)

    def lcs_ratio_betaplot():
        # tells us beta = 10 is decent?
        fig, [mu0ax, mu100ax] = plt.subplots(1, 2, figsize=(9, 3), sharey=True)

        for (mu, ax) in [(0, mu0ax), (100, mu100ax)]:
            df = lcs_df_2.xs('lcsratio', level='feature_structures')
            df = df.xs(mu, level='mu')
            df = df.reset_index()
            df = df[df.trainsize > 10]
            
            means = df.pivot_table(index='beta', columns='trainsize', values='fmeasure', aggfunc=np.mean)
            means.plot(ax=ax, marker='o', markersize=2)
            
            ax.legend(loc='lower left', title='train size', prop={'size':7})
            ax.set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
            ax.set_xlabel("$\\beta$", labelpad=0)
            ax.set(ylabel='$F_1$ measure')
            ax.grid(True)
        
        mu0ax.set(title="Choice of $\\beta$ for $\\mu=0$")
        mu100ax.set(title="Choice of $\\beta$ for $\\mu=100$")

        fig.tight_layout()
        fig.savefig('/tmp/lcs_beta_plot.pdf', format='pdf', dpi=1000)

    lcs_ratio_monsterplot()
    lcs_ratio_betaplot()


pos_gold_plots(relative=True)
#lcs_plots()
