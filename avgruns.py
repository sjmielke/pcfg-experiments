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
    indices = ['trainsize','feature_structures','oov_handling','eta']
    
    pg_df = join_file_frames([logroot + "/pos_trainsize_eta.log", logroot + "/pos_trainsize_eta_other.log"], indices)
    
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    pg_df = pg_df.xs('postagsonly', level='feature_structures')
    pg_df = pg_df.reset_index()
    rootdata = pg_df[pg_df.trainsize > 10]
    rootdata = rootdata[rootdata.eta > 0]

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

            means = pg_df.pivot_table(index='eta', columns='trainsize', values='fmeasure', aggfunc=np.mean)
            errs = pg_df.pivot_table(index='eta', columns='trainsize', values='fmeasure', aggfunc=np.std)
            
            means = means.rolling(window=smoothify, min_periods=1).mean()
            
            if relative:
                means = means.apply(lambda ser: ser / ser[0.001])
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

            #axes[j][i].set_xscale('symlog', linthreshx=0.001, linscalex=0.2)
            axes[j][i].set_xlabel("$\\eta$", labelpad=0)
            axes[j][i].set(ylabel="$F_1$ measure" + " (relative from $\\eta$=0)" if relative else "", title="Choice of $\\eta$ \\small ({})".format(title))
            axes[j][i].grid(True)

    fig.tight_layout()
    fig.savefig('/tmp/pos_eta_plot.pdf', format='pdf', dpi=1000)
    fig.savefig('/tmp/pos_eta_plot.png', format='png', dpi=1000)


def lcs_plots():
    indices = ['trainsize','feature_structures','oov_handling','eta','beta']

    lcs_df = join_file_frames([logroot + "/lcs_trainsize_beta_eta.log", logroot + "/lcs_trainsize_beta_eta_39k.log"], indices)

    tss = [10,50,100,500,1000,5000,10000,20000,39000]

    def lcs_ratio_monsterplot():
        fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True) #, sharey=True)

        for j in range(3):
            for i in range(3):
                ts = tss[i + 3*j]
                
                if (lcs_df.index.get_level_values('trainsize') == ts).any():
                    tsdf = lcs_df.xs(ts, level='trainsize')
                    lcs_only_df = tsdf.xs('lcsratio', level='feature_structures')
                    
                    mask = lcs_only_df.index.get_level_values('beta').isin([1.0, 4.0, 10.0, 20.0, 30.0, 50.0])
                    lcs_only_df = lcs_only_df[mask].reset_index()
                    
                    means = lcs_only_df.pivot_table(index='eta', columns='beta', values='fmeasure', aggfunc=np.mean)
                    errs = lcs_only_df.pivot_table(index='eta', columns='beta', values='fmeasure', aggfunc=np.std)
                    try:
                        baseline = lcs_df.loc[ts].loc["exactmatch"].loc["uniform"].loc[1.0].loc[1.0]
                        baseline = baseline['fmeasure']
                        baseline = np.mean(baseline)
                        
                        means.loc[0.0] = baseline
                        means = means.sort_index()
                        
                        means.plot(ax=ax[j][i], marker='o', markersize=2, yerr=errs)
                        
                        ax[j][i].axhline(baseline, 0.0, 1.0, linestyle='dashed', linewidth=1, color='black')
                    except:
                        pass

                ax[j][i].set_xscale('symlog', linthreshx=0.001, linscalex=0.2)
                ax[j][i].set_xlabel("$\\eta$", labelpad=0)
                (bot, top) = ax[j][i].get_ylim()
                ax[j][i].set_ylim(top - (top-bot)/3, top)
                ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
                ax[j][i].legend(loc='lower right', title='$\\beta$', prop={'size':7})
                ax[j][i].grid(True)

        fig.tight_layout()
        fig.savefig('/tmp/lcs_ratio_monsterplot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/lcs_ratio_monsterplot_eta.png', format='png', dpi=1000)

    def lcs_ratio_betaplot():
        # tells us beta = 10 is decent?
        fig, [eta0ax, eta100ax] = plt.subplots(1, 2, figsize=(9, 3), sharey=True)

        for (eta, ax) in [(0.005, eta0ax), (0.5, eta100ax)]:
            df = lcs_df.xs('lcsratio', level='feature_structures')
            try:
                df = df.xs(eta, level='eta')
                df = df.reset_index()
                #df = df[df.trainsize > 10]
                
                means = df.pivot_table(index='beta', columns='trainsize', values='fmeasure', aggfunc=np.mean)
                means.plot(ax=ax, marker='o', markersize=2)
                
                ax.legend(loc='lower left', title='train size', prop={'size':7})
                ax.set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
                ax.set_xlabel("$\\beta$", labelpad=0)
                ax.set(ylabel='$F_1$ measure')
                ax.grid(True)
                ax.set(title="Choice of $\\beta$ for $\\eta={}$".format(eta))
            except:
                pass

        fig.tight_layout()
        fig.savefig('/tmp/lcs_beta_plot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/lcs_beta_plot_eta.png', format='png', dpi=1000)

    lcs_ratio_monsterplot()
    lcs_ratio_betaplot()


def dice_plots():
    indices = ['trainsize','feature_structures','oov_handling','eta','kappa']

    dice_df = join_file_frames([logroot + "/dice_trainsize_kappa_eta.log", logroot + "/dice_trainsize_hugekappa_eta.log", logroot + "/dice_trainsize_kappa_eta_39k.log"], indices)

    tss = [10,50,100,500,1000,5000,10000,20000,39000]

    def dice_monsterplot():
        fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True) #, sharey=True)

        for j in range(3):
            for i in range(3):
                ts = tss[i + 3*j]
                
                if (dice_df.index.get_level_values('trainsize') == ts).any():
                    tsdf = dice_df.xs(ts, level='trainsize')
                    dice_only_df = tsdf.xs('dice', level='feature_structures')
                    
                    mask = dice_only_df.index.get_level_values('kappa').isin([1, 2, 4, 6, 8, 10, 50, 100])
                    dice_only_df = dice_only_df[mask].reset_index()
                    
                    means = dice_only_df.pivot_table(index='eta', columns='kappa', values='fmeasure', aggfunc=np.mean)
                    errs = dice_only_df.pivot_table(index='eta', columns='kappa', values='fmeasure', aggfunc=np.std)
                    try:
                        baseline = dice_df.loc[ts].loc["exactmatch"].loc["uniform"].loc[1.0].loc[1.0]
                        baseline = baseline['fmeasure']
                        baseline = np.mean(baseline)
                        
                        means.loc[0.0] = baseline
                        means = means.sort_index()
                        
                        means.plot(ax=ax[j][i], marker='o', markersize=2, yerr=errs)
                        
                        ax[j][i].axhline(baseline, 0.0, 1.0, linestyle='dashed', linewidth=1, color='black')
                    except:
                        pass

                ax[j][i].set_xscale('symlog', linthreshx=0.001, linscalex=0.2)
                ax[j][i].set_xlabel("$\\eta$", labelpad=0)
                (bot, top) = ax[j][i].get_ylim()
                ax[j][i].set_ylim(top - (top-bot)/2, top)
                ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
                ax[j][i].legend(loc='lower right', title='$\\kappa$', prop={'size':7})
                ax[j][i].grid(True)

        fig.tight_layout()
        fig.savefig('/tmp/dice_monsterplot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/dice_monsterplot_eta.png', format='png', dpi=1000)

    def dice_kappaplot():
        # tells us kappa = 10 is decent?
        fig, [eta0ax, eta100ax] = plt.subplots(1, 2, figsize=(9, 3), sharey=True)

        for (eta, ax) in [(0.005, eta0ax), (0.5, eta100ax)]:
            df = dice_df.xs('dice', level='feature_structures')
            try:
                df = df.xs(eta, level='eta')
                df = df.reset_index()
                #df = df[df.trainsize > 10]
                
                means = df.pivot_table(index='kappa', columns='trainsize', values='fmeasure', aggfunc=np.mean)
                means.plot(ax=ax, marker='o', markersize=2)
                
                ax.legend(loc='upper left', title='train size', prop={'size':7})
                #ax.set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
                ax.set_xscale('log')
                ax.set_xlabel("$\\kappa$", labelpad=0)
                ax.set(ylabel='$F_1$ measure')
                ax.grid(True)
                ax.set(title="Choice of $\\kappa$ for $\\eta={}$".format(eta))
            except:
                pass

        fig.tight_layout()
        fig.savefig('/tmp/dice_kappa_plot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/dice_kappa_plot_eta.png', format='png', dpi=1000)

    dice_monsterplot()
    dice_kappaplot()


#pos_gold_plots(relative=True)
#lcs_plots()
dice_plots()
