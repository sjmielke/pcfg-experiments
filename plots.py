import itertools
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif=['charter'])
markers = ['v','^','<','>','X','o','D','d']

logroot = "/home/sjm/documents/Uni/FuzzySP/pcfg-experiments/logs/"
tss = [100,500,1000,5000,10000,40472]

def join_file_frames(filenames, indices):
    df = pd.DataFrame()
    for filename in filenames:
        df = df.append(pd.DataFrame.from_csv(filename, sep='\t', header=0, index_col=indices).applymap(lambda x: round(x, 7) if isinstance(x, float) else x))

    return df #.sort_index(axis=0, level=indices, sort_remaining=False)

def six_facet_plot(feature_structure, filenames, series_names, columnmapper = None, ignore_multiplies = False, legend_title = None, ywindow_size = None, ywindow_mid = lambda x, y: x + y / 2):
    indices = ['trainsize','feature_structures','oov_handling','eta','beta']
    indices += [sn for sn in series_names if sn not in indices]
    
    df = join_file_frames(filenames, indices).xs(feature_structure, level='feature_structures')
    
    fig, ax = plt.subplots(2, 3, figsize=(10, 7)) #, sharex=True) #, sharey=True)

    for j in range(2):
        for i in range(3):
            ts = tss[i + 3*j]
            
            if (df.index.get_level_values('trainsize') == ts).any():
                ts_df = df.loc[ts] # df.xs(ts, level='trainsize')
                ts_df = ts_df.reset_index()
                
                def assert_singleton(nda):
                    if nda.size > 1:
                        if not ignore_multiplies:
                            print(f"Aggregating in ts {ts}: ", nda)
                        return np.mean(nda)
                    else:
                        return nda
                
                # First pivot
                plot_df = ts_df.pivot_table(index='eta', columns=series_names, values='fmeasure', aggfunc=assert_singleton)
                
                # Simplify and potentially combine columns
                if columnmapper:
                    plot_df.columns = plot_df.columns.map(columnmapper)
                    plot_df.sort_index(axis=1, inplace=True)
                    cm = columnmapper
                else:
                    def cm(x):
                        [v] = x
                        return v
                
                series_indices = [df.index.names.index(sn) for sn in series_names]
                
                # Make sure all series exist in each subplot
                all_series = [cm([t[i] for i in series_indices]) for t in df.index]
                # OrderedDict is just an OrderedSet (values = None, are ignored)
                all_series = OrderedDict.fromkeys(all_series)
                all_series = list(all_series)
                plot_df = plot_df.reindex(columns=all_series)
                
                # Now plot each w/ different marker
                for (series, marker) in zip(plot_df.columns, itertools.cycle(markers)):
                    plot_df[series].plot(ax=ax[j][i], marker=marker, markersize=3)
                
                #plot_df.plot(ax=ax[j][i], marker='o', markersize=2)
            
            ax[j][i].set_xscale('symlog', linthreshx=0.001, linscalex=0.2)
            ax[j][i].set_xlabel("$\\eta$", labelpad=0)
            if ywindow_size:
                (bot, top) = ax[j][i].get_ylim()
                mid = ywindow_mid(bot, top)
                ax[j][i].set_ylim(mid - ywindow_size, mid + ywindow_size)
            ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
            ax[j][i].grid(True)

    ax[1][1].legend(title=legend_title, loc='center', prop={'size':9}, bbox_to_anchor=(0.5, -0.275), ncol=len(all_series) + 1)


    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig




def simplify_postags_file(s):
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
        return "trainsize" + ", " + s[1][0] + "-best"

six_facet_plot('lcsratio',
    filenames = [logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"],
    series_names = ['beta'],
    legend_title = '\\beta',
    ywindow_size = 7,
    ywindow_mid = lambda bot, top: top - 7
    ).savefig('/tmp/lcsratio_monsterplot_eta.pdf', format='pdf', dpi=1000)

six_facet_plot('postagsonly',
    filenames = [logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"],
    series_names = ['testtagsfile','nbesttags'],
    columnmapper = simplify_postags_file,
    ignore_multiplies = True, # all tags when ts=all...
    legend_title = 'tagger trained on'
    ).savefig('/tmp/tagged_monsterplot_eta.pdf', format='pdf', dpi=1000)

six_facet_plot('levenshtein',
    filenames = [logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"],
    series_names = ['beta'],
    legend_title = '\\beta',
    ywindow_size = 7,
    ywindow_mid = lambda bot, top: top - 7
    ).savefig('/tmp/levenshtein_monsterplot_eta.pdf', format='pdf', dpi=1000)
