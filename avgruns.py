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

    return df #.sort_index(axis=0, level=indices, sort_remaining=False)

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

def tagged_plots():
    indices = ['trainsize','feature_structures','oov_handling','eta','testtagsfile','nbesttags']
    #indices = ['trainsize','feature_structures','oov_handling','eta','nbesttags']

    tagged_df = join_file_frames([logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"], indices).xs('postagsonly', level='feature_structures')

    tss = [100,500,1000,5000,10000,40472]

    def tagged_monsterplot():
        fig, ax = plt.subplots(2, 3, figsize=(10, 6)) #, sharex=True) #, sharey=True)

        for j in range(3):
            for i in range(3):
                if len(tss) <= i + 3*j:
                    continue
                ts = tss[i + 3*j]
                
                if (tagged_df.index.get_level_values('trainsize') == ts).any():
                    #tsdf = tagged_df.sort_index()
                    #tsdf = tsdf.groupby(tsdf.index).max()
                    #tsdf.index = pd.MultiIndex.from_tuples(tsdf.index, names=indices)
                    tagged_only_df = tagged_df.loc[ts] # tagged_df.xs(ts, level='trainsize')
                    tagged_only_df = tagged_only_df.reset_index()
                    
                    means = tagged_only_df.pivot_table(index='eta', columns=['testtagsfile','nbesttags'], values='fmeasure', aggfunc=np.mean)
                    #means = tagged_only_df.pivot_table(index='eta', columns='nbesttags', values='fmeasure', aggfunc=np.mean)
                    
                    def simplifyfile(s):
                        if '.' not in s[0]:
                            assert(s[1] == "1besttags")
                            return "gold tags" if s[0] == "" else s[0]
                        l = s[0].split(".")
                        mode = l[-1]
                        sts = l[-2]
                        if mode == "gold":
                            return "gold tags"
                        elif sts == "39000":
                            return "all (39000), " + s[1][0] + "-best"
                        elif sts == "40472":
                            return "all (40472), " + s[1][0] + "-best"
                        else:
                            return sts + ", " + s[1][0] + "-best"
                    
                    means.columns = means.columns.map(simplifyfile)
                    means.sort_index(axis=1, inplace=True)
                    
                    
                    means.plot(ax=ax[j][i], marker='o', markersize=2)
                
                ax[j][i].set_xscale('symlog', linthreshx=0.01, linscalex=0.3)
                ax[j][i].set_xlabel("$\\eta$", labelpad=0)
                # if ts > 100:
                #     (bot, top) = ax[j][i].get_ylim()
                #     mid = (bot + top) / 2
                #     ax[j][i].set_ylim(mid - 7, mid + 7)
                ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
                ax[j][i].legend(loc='lower right', title='tagger from', prop={'size':7})
                ax[j][i].grid(True)

        fig.tight_layout()
        fig.savefig('/tmp/tagged_monsterplot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/tagged_monsterplot_eta.png', format='png', dpi=1000)
    
    tagged_monsterplot()

def lcs_plots():
    indices = ['trainsize','feature_structures','oov_handling','eta','beta']

    lcs_df = join_file_frames([logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"], indices)

    tss = [100,500,1000,5000,10000,40472]

    def lcs_ratio_monsterplot():
        fig, ax = plt.subplots(2, 3, figsize=(10, 6)) #, sharex=True) #, sharey=True)

        for j in range(3):
            for i in range(3):
                if len(tss) <= i + 3*j:
                    continue
                ts = tss[i + 3*j]
                
                if (lcs_df.index.get_level_values('trainsize') == ts).any():
                    lcs_only_df = lcs_df.xs('lcsratio', level='feature_structures')
                    lcs_only_df = lcs_only_df.xs(ts, level='trainsize')
                    
                    #mask = lcs_only_df.index.get_level_values('beta').isin([1.0, 4.0, 10.0, 20.0, 30.0, 50.0])
                    #lcs_only_df = lcs_only_df[mask].reset_index()
                    lcs_only_df = lcs_only_df.reset_index()
                    
                    means = lcs_only_df.pivot_table(index='eta', columns='beta', values='fmeasure', aggfunc=np.mean)
                    errs = lcs_only_df.pivot_table(index='eta', columns='beta', values='fmeasure', aggfunc=np.std)
                    try:
                        # baseline = lcs_df.loc[ts].loc["exactmatch"].loc["uniform"].loc[1.0].loc[1.0]
                        # baseline = baseline['fmeasure']
                        # baseline = np.mean(baseline)
                        # 
                        # means.loc[0.0] = baseline
                        # means = means.sort_index()
                        
                        means.plot(ax=ax[j][i], marker='o', markersize=2, yerr=errs)
                        
                        ax[j][i].axhline(baseline, 0.0, 1.0, linestyle='dashed', linewidth=1, color='black')
                    except:
                        pass

                ax[j][i].set_xscale('symlog', linthreshx=0.001, linscalex=0.2)
                ax[j][i].set_xlabel("$\\eta$", labelpad=0)
                #(bot, top) = ax[j][i].get_ylim()
                #ax[j][i].set_ylim(top - (top-bot)/3, top)
                ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
                ax[j][i].legend(loc='lower right', title='$\\beta$', prop={'size':7})
                ax[j][i].grid(True)

        fig.tight_layout()
        fig.savefig('/tmp/lcs_ratio_monsterplot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/lcs_ratio_monsterplot_eta.png', format='png', dpi=1000)

    def lcs_ratio_betaplot():
        # tells us beta = 10 is decent?
        fig, [eta0ax, eta50ax, eta100ax] = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

        for (eta, ax) in [(0.006, eta0ax), (0.06, eta50ax), (0.6, eta100ax)]:
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

    def lcs_ratio_alphaplot():
        df = join_file_frames([logroot + "/german_lcs_alpha.log", logroot + "/german_lcs_alpha_apocalypse.log"], indices)
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        
        try:
            df = df.reset_index()
            #df = df[df.trainsize > 10]
            
            means = df.pivot_table(index='alpha', columns='trainsize', values='fmeasure', aggfunc=np.mean)
            means.plot(ax=ax, marker='o', markersize=2)
            
            ax.legend(loc='lower right', title='train size', prop={'size':7})
            #ax.set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
            ax.set_xlabel("$\\alpha$", labelpad=0)
            ax.set(ylabel='$F_1$ measure')
            ax.grid(True)
        except:
            pass

        fig.tight_layout()
        fig.savefig('/tmp/lcs_alpha_plot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/lcs_alpha_plot_eta.png', format='png', dpi=1000)

    lcs_ratio_monsterplot()
    lcs_ratio_betaplot()
    lcs_ratio_alphaplot()

def dice_plots():
    indices = ['trainsize','feature_structures','oov_handling','eta','kappa']

    dice_df = join_file_frames([logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"], indices)

    tss = [100,500,1000,5000,10000,40472]

    def dice_monsterplot():
        fig, ax = plt.subplots(2, 3, figsize=(10, 6)) #, sharex=True) #, sharey=True)

        for j in range(3):
            for i in range(3):
                if len(tss) <= i + 3*j:
                    continue
                ts = tss[i + 3*j]
                
                if (dice_df.index.get_level_values('trainsize') == ts).any():
                    dice_only_df = dice_df.xs('dice', level='feature_structures')
                    dice_only_df = dice_only_df.xs(ts, level='trainsize')
                    
                    mask = dice_only_df.index.get_level_values('kappa').isin([1, 3, 5, 10, 50])
                    dice_only_df = dice_only_df[mask].reset_index()
                    #dice_only_df = dice_only_df.reset_index()
                    
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
        fig, [eta0ax, eta50ax, eta100ax] = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

        for (eta, ax) in [(0.006, eta0ax), (0.06, eta50ax), (0.6, eta100ax)]:
            df = dice_df.xs('dice', level='feature_structures')
            try:
                df = df.xs(eta, level='eta')
                df = df.reset_index()
                #df = df[df.trainsize > 10]
                
                means = df.pivot_table(index='kappa', columns='trainsize', values='fmeasure', aggfunc=np.mean)
                means.plot(ax=ax, marker='o', markersize=2)
                
                ax.legend(loc='lower right', title='train size', prop={'size':7})
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

def levenshtein_plots():
    indices = ['trainsize','feature_structures','oov_handling','eta','beta']

    levenshtein_df = join_file_frames([logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"], indices)

    tss = [100,500,1000,5000,10000,40472]

    def levenshtein_monsterplot():
        fig, ax = plt.subplots(2, 3, figsize=(10, 6)) #, sharex=True) #, sharey=True)

        for j in range(3):
            for i in range(3):
                if len(tss) <= i + 3*j:
                    continue
                ts = tss[i + 3*j]
                
                if (levenshtein_df.index.get_level_values('trainsize') == ts).any():
                    levenshtein_only_df = levenshtein_df.xs('levenshtein', level='feature_structures')
                    levenshtein_only_df = levenshtein_only_df.xs(ts, level='trainsize')
                    
                    #mask = levenshtein_only_df.index.get_level_values('beta').isin([1.0, 4.0, 10.0, 20.0, 30.0, 50.0])
                    #levenshtein_only_df = levenshtein_only_df[mask].reset_index()
                    levenshtein_only_df = levenshtein_only_df.reset_index()
                    
                    means = levenshtein_only_df.pivot_table(index='eta', columns='beta', values='fmeasure', aggfunc=np.mean)
                    errs = levenshtein_only_df.pivot_table(index='eta', columns='beta', values='fmeasure', aggfunc=np.std)
                    try:
                        baseline = levenshtein_df.loc[ts].loc["exactmatch"].loc["uniform"].loc[1.0].loc[1.0]
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
                    ax[j][i].set_ylim(min(baseline - 1.0, top - (top-bot)/3), top)
                    ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
                    ax[j][i].legend(loc='lower right', title='$\\beta$', prop={'size':7})
                    ax[j][i].grid(True)

        fig.tight_layout()
        fig.savefig('/tmp/levenshtein_monsterplot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/levenshtein_monsterplot_eta.png', format='png', dpi=1000)

    def levenshtein_betaplot():
        # tells us beta = 10 is decent?
        fig, [eta0ax, eta50ax, eta100ax] = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

        for (eta, ax) in [(0.006, eta0ax), (0.06, eta50ax), (0.6, eta100ax)]:
            df = levenshtein_df.xs('levenshtein', level='feature_structures')
            try:
                df = df.xs(eta, level='eta')
                df = df.reset_index()
                #df = df[df.trainsize > 10]
                
                means = df.pivot_table(index='beta', columns='trainsize', values='fmeasure', aggfunc=np.mean)
                means.plot(ax=ax, marker='o', markersize=2)
                
                ax.legend(loc='lower right', title='train size', prop={'size':7})
                ax.set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
                ax.set_xlabel("$\\beta$", labelpad=0)
                ax.set(ylabel='$F_1$ measure')
                ax.grid(True)
                ax.set(title="Choice of $\\beta$ for $\\eta={}$".format(eta))
            except:
                pass

        fig.tight_layout()
        fig.savefig('/tmp/levenshtein_beta_plot_eta.pdf', format='pdf', dpi=1000)
        fig.savefig('/tmp/levenshtein_beta_plot_eta.png', format='png', dpi=1000)

    levenshtein_monsterplot()
    levenshtein_betaplot()

def ml_tagged_plots(noafterdash):
    indices = ['language', 'trainsize','feature_structures','oov_handling','eta','noafterdash']

    ml_tagged_df = join_file_frames([logroot + f"/multilang_tagged_trainsize_eta.log"], indices)
    ml_tagged_df = ml_tagged_df.xs(noafterdash, level='noafterdash')

    tss = [100,500,1000,5000,10,-1]

    def ml_tagged_monsterplot():
        fig, ax = plt.subplots(2, 3, figsize=(8, 8)) #, sharex=True, sharey=True)

        for j in range(2):
            for i in range(3):
                ts = tss[i + 3*j]
                
                if (ml_tagged_df.index.get_level_values('trainsize') == ts).any():
                    tsdf = ml_tagged_df.xs(ts, level='trainsize')
                    
                    ml_tagged_only_df = tsdf.xs('postagsonly', level='feature_structures').reset_index()
                    
                    means = ml_tagged_only_df.pivot_table(index='eta', columns='language', values='fmeasure', aggfunc=np.mean)
                    
                    from collections import OrderedDict
                    all_langs = list(OrderedDict.fromkeys([t[0] for t in ml_tagged_df.index if t[1] >= ts]))
                    means = means.reindex(columns=all_langs)
                    
                    means = means.sort_index(axis=0, level='eta', sort_remaining=False)
                    
                    for (lang, marker) in zip(all_langs, ['v','^','<','>','X','o','D','d']):
                        means[lang].plot(ax=ax[j][i], marker=marker, markersize=3)
                
                ax[j][i].set_xscale('symlog', linthreshx=0.001, linscalex=0.3)
                ax[j][i].set_xlabel("$\\eta$", labelpad=0)
                # if ts > 100:
                #     (bot, top) = ax[j][i].get_ylim()
                #     mid = (bot + top) / 2
                #     ax[j][i].set_ylim(mid - 7, mid + 7)
                ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
                ax[j][i].grid(True)
        
        ax[1][1].legend(loc='center', prop={'size':8}, bbox_to_anchor=(0.5, -0.2), ncol=len(all_langs))

        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(f'/tmp/ml_tagged_monsterplot_eta_{noafterdash}.pdf', format='pdf', dpi=1000)
        #fig.savefig(f'/tmp/ml_tagged_monsterplot_eta_{noafterdash}.png', format='png', dpi=1000)
    
    ml_tagged_monsterplot()

#pos_gold_plots(relative=True)
tagged_plots()
lcs_plots()
dice_plots()
levenshtein_plots()

#ml_tagged_plots('nt_as_is')
#ml_tagged_plots('noafterdash')
