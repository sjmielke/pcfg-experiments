import itertools
import sys
from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif=['charter'])
markers = ['v','^','<','>','X','o','D','d']

logroot = "/home/sjm/documents/Uni/FuzzySP/pcfg-experiments/logs/"

def join_file_frames(filenames, indices):
    df = pd.DataFrame()
    for filename in filenames:
        df = df.append(pd.DataFrame.from_csv(filename, sep='\t', header=0, index_col=indices).applymap(lambda x: round(x, 7) if isinstance(x, float) else x))

    return df #.sort_index(axis=0, level=indices, sort_remaining=False)

def assert_singleton(nda):
    if nda.size > 1:
        print(f"Aggregating in ts {ts}: ", nda)
        return np.mean(nda)
    else:
        return nda

def multi_facet_plot(
    feature_structure,
    filenames,
    series_names,
    x_name = 'eta',
    x_title = '$\\eta$',
    df_restricter = None,
    columnmapper = None,
    aggregator = np.mean,
    legend_title = None,
    legend_ncols = None,
    ywindow_size = None,
    ywindow_mid = lambda x, y: x + y / 2,
    facet_name = 'trainsize',
    facets = [100,500,1000,5000,10000,40472],
    nrows = 2):
    
    indices = ['trainsize','feature_structures','oov_handling','eta','beta']
    indices += [sn for sn in series_names if sn not in indices]
    
    df = join_file_frames(filenames, indices).xs(feature_structure, level='feature_structures')
    
    if df_restricter:
        df = df_restricter(df)
    
    if not facet_name:
        facets = ['']
    
    if not legend_title:
        legend_title = ' $\\times$ '.join(series_names)
    if not x_title:
        x_title = x_name
    
    ncols = int(ceil(len(facets)/nrows))
    
    if facet_name:
        xsize, ysize = 3*ncols, 3*nrows + 0.45
    else:
        xsize, ysize = 5, 3.75
    
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(xsize, ysize)) #, sharex=True) #, sharey=True)

    for j in range(nrows):
        for i in range(ncols):
            facet = facets[i + ncols*j]
            
            if not facet_name or (df.index.get_level_values(facet_name) == facet).any():
                if facet_name:
                    facet_df = df.loc[facet] # df.xs(facet, level='trainsize')
                else:
                    facet_df = df
                facet_df = facet_df.reset_index()
                
                # First pivot
                plot_df = facet_df.pivot_table(index=x_name, columns=series_names, values='fmeasure', aggfunc=aggregator)
                
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
            
            if x_name == 'eta':
                ax[j][i].set_xscale('symlog', linthreshx=0.001, linscalex=0.2)
            ax[j][i].set_xlabel(x_title, labelpad=0)
            if ywindow_size:
                (bot, top) = ax[j][i].get_ylim()
                mid = ywindow_mid(bot, top)
                ax[j][i].set_ylim(mid - ywindow_size, mid + ywindow_size)
            ax[j][i].set(ylabel='$F_1$ measure')
            if facet_name:
                ax[j][i].set(title=f"{facet_name} {facet}")
            ax[j][i].grid(True)
    
    fig.legend(ax[0][0].get_lines(), all_series, 'lower center', title=legend_title, ncol=legend_ncols if legend_ncols else len(all_series), fontsize=9 if facet_name else 7)

    fig.tight_layout(rect=[0, 0.05/nrows + 0.05, 1, 1])
    return fig



def restricter_and(df, **kvargs):
    for k, v in kvargs.items():
        df = df[df[k] == v]
    return df

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

multi_facet_plot('postagsonly',
    filenames = [logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"],
    series_names = ['testtagsfile','nbesttags'],
    columnmapper = simplify_postags_file,
    aggregator = np.mean, # all tags when ts=all...
    legend_title = 'tagger trained on'
    ).savefig('/tmp/tagged_monsterplot_eta.pdf', format='pdf', dpi=1000)

multi_facet_plot('lcsratio',
    filenames = [logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"],
    series_names = ['beta'],
    legend_title = '$\\beta$',
    ywindow_size = 7,
    ywindow_mid = lambda bot, top: top - 7    ).savefig('/tmp/lcsratio_monsterplot_eta.pdf', format='pdf', dpi=1000)

multi_facet_plot('lcsratio',
    filenames = [logroot + "/german_06-10_lcsalphatune.log"],
    series_names = ['trainsize'],
    x_name = 'alpha',
    x_title = '$\\alpha$',
    
    facet_name = None,
    nrows = 1
    ).savefig('/tmp/lcsratio_alphaplot_.pdf', format='pdf', dpi=1000)

multi_facet_plot('levenshtein',
    filenames = [logroot + "/german_megatune.log", logroot + "/german_apocalypsetune_coarse.log"],
    series_names = ['beta'],
    legend_title = '$\\beta$',
    ywindow_size = 7,
    ywindow_mid = lambda bot, top: top - 7
    ).savefig('/tmp/levenshtein_monsterplot_eta.pdf', format='pdf', dpi=1000)

multi_facet_plot('ngrams',
    filenames = [logroot + "/german_06-10_omnitune.log"],
    series_names = ['kappa','beta'],
    df_restricter = lambda df: restricter_and(df, dualmono_pad = 'fullpad'),
    columnmapper = lambda t: f"$\\kappa={t[0]}, \\beta={t[1]}$",
    legend_title = '$\\kappa \\times \\beta$',
    legend_ncols = 5,
    ywindow_size = 7,
    ywindow_mid = lambda bot, top: top - 7
    ).savefig('/tmp/ngrams_omni_monsterplot_eta.pdf', format='pdf', dpi=1000)

for kappa in [1, 2, 3, 5, 10]:
    multi_facet_plot('ngrams',
        filenames = [logroot + "/german_06-10_omnitune.log", logroot + "/german_06-11_eta0001.log"],
        series_names = ['beta'],
        df_restricter = lambda df: restricter_and(df, dualmono_pad = 'fullpad', kappa = kappa),
        legend_title = '$\\beta$',
        ywindow_size = 7,
        ywindow_mid = lambda bot, top: top - 7,
        facets = [100,1000,10000,40472],
        nrows = 1
        ).savefig(f"/tmp/ngrams_kappa{kappa}_monsterplot_eta.pdf", format='pdf', dpi=1000)
