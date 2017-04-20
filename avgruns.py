import sys
import pandas as pd
import matplotlib.pyplot as plt

######################### Get data ##########################

def single_file(filename):
    nooffiles = None
    df = pd.DataFrame()

    with open(filename) as f:
        lines = f.read().splitlines()

    for line in lines[1:]:
        valme = []
        for field in line.split('\t'):
            # Try casting!
            try:
                valme.append(int(field))
            except ValueError:
                try:
                    valme.append(float(field))
                except ValueError:
                    valme.append(field)
        
        # Enter values into data frame
        new_df = pd.DataFrame([valme], columns=lines[0].split('\t'))
        df = df.append(new_df)
    
    return df.set_index(['trainsize','feature_structures','oov_handling','mu','beta'])

def average_files(filenames):
    nooffiles = None
    df = pd.DataFrame()

    lines = []
    for fn in filenames:
        with open(fn) as f:
            lines.append(f.read().splitlines())
    alllines = list(zip(*lines))

    for variants in alllines[1:]:
        variants = [v.split('\t') for v in variants]
        #printme = []
        valme = []
        for fieldvals in zip(*variants):
            if nooffiles == None:
                nooffiles = len(fieldvals)
            assert(len(fieldvals) == nooffiles)
            if len(set(fieldvals)) == 1:
                #printme.append(fieldvals[0])
                # Try casting!
                try:
                    valme.append(int(fieldvals[0]))
                except ValueError:
                    try:
                        valme.append(float(fieldvals[0]))
                    except ValueError:
                        valme.append(fieldvals[0])
            else:
                vals = [float(v) for v in fieldvals]
                avg = round(sum(vals) / len(vals), 7)
                #printme.append("{:.3f}".format(avg))
                valme.append(avg)
        #print('\t'.join(printme))
        
        # Enter values into data frame
        heads = None
        for l in alllines[0]:
            heads = heads or l
            assert(heads == l)
        new_df = pd.DataFrame([valme], columns=heads.split('\t'))
        df = df.append(new_df, ignore_index=True)
        
    print("{} files averaged.".format(nooffiles), file=sys.stderr)
    return  df.set_index(['trainsize','feature_structures','oov_handling','mu','beta'])

######################### Crunch data ##########################

vara = pd.DataFrame.from_csv('/tmp/t/lcsratio_trainsize_beta_mu_1.log', sep='\t', header=0, index_col=['trainsize','feature_structures','oov_handling','mu','beta']).applymap(lambda x: round(x, 7) if isinstance(x, float) else x)






print(vara.xs('lcsratio', level='feature_structures'))
#exit(0)








varb = single_file("/tmp/t/lcsratio_trainsize_beta_mu_1.log")

vara.to_csv('/tmp/old.csv')
varb.to_csv('/tmp/new.csv')






adf = average_files(["/tmp/t/lcsratio_trainsize_beta_mu_{}.log".format(i) for i in range(1,9)])

def average_frames(dfs):
    def coladd(c1,c2):
        if c1.dtype == object or c2.dtype == object:
            assert((c1 == c2).all())
            return c1
        else:
            return c1+c2
    def coldiv(c):
        if isinstance(c, str):
            return c
        elif isinstance(c, int):
            return c // len(dfs)
        else:
            return round(c / len(dfs), 7)
    
    d1 = dfs[0].copy()
    for d2 in dfs[1:]:
        d1 = d1.combine(d2, coladd)
    
    return d1.applymap(coldiv)

dfs = []
for i in range(1,9):
    dfs.append(pd.DataFrame.from_csv("/tmp/t/lcsratio_trainsize_beta_mu_{}.log".format(i), sep='\t', header=0, index_col=['trainsize','feature_structures','oov_handling','mu','beta']).applymap(lambda x: round(x, 7) if isinstance(x, float) else x))

aadf = average_frames(dfs)

adf.to_csv('/tmp/old.csv')
aadf.to_csv('/tmp/new.csv')

print(adf.equals(aadf))











df = vara.xs('lcsratio', level='feature_structures')












#df = adf[adf.feature_structures == 'lcsratio']

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
            tsdf = tsdf.pivot(index='mu', columns='beta', values='fmeasure')
            tsdf.plot(ax=ax[j][i], marker='o', markersize=2)

        ax[j][i].set_xscale('symlog', linthreshx=0.1, linscalex=0.2)
        ax[j][i].set_xlabel("$\\mu$", labelpad=0)
        ax[j][i].set(ylabel='$F_1$ measure', title="trainsize {}".format(ts))
        ax[j][i].legend(loc='lower right', title='$\\beta$', prop={'size':7})
        ax[j][i].grid(True)


fig.tight_layout()
fig.savefig('/tmp/test.pdf', format='pdf', dpi=1000)










