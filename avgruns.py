import sys

lines = []

nooffiles = None

for fn in sys.argv[1:]:
    with open(fn) as f:
        lines.append(f.read().splitlines())

for variants in zip(*lines):
    variants = [v.split('\t') for v in variants]
    printme = []
    for fieldvals in zip(*variants):
        if nooffiles == None:
            nooffiles = len(fieldvals)        
        assert(len(fieldvals) == nooffiles)
        if len(set(fieldvals)) == 1:
            printme.append(fieldvals[0])
        else:
            vals = [float(v) for v in fieldvals]
            printme.append("{:.3f}".format(sum(vals) / len(vals)))
    print('\t'.join(printme))

print("{} files averaged.".format(nooffiles), file=sys.stderr)