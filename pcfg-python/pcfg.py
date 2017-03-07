import itertools
from collections import defaultdict, Counter
import math

from nltk.corpus import treebank

import timeit
import resource

import multiprocessing

import json

# Rule :: (int/LHS, [int/RHS or str/terminal])
# rules :: {int/LHS: [([int/RHS or str/terminal], float/prob)]}

def print_grammar(prules, ntdict):
	print("{} NTs, {} rules:".format(len(ntdict), len(list(itertools.chain(*prules.values())))))
	
	#print(ntdict)
	return
	
	for (lhs, rhss) in prules.items():
		for (rhs, prob) in rhss:
			def print_rhs_part(p):
				if isinstance(p, int):
					return ntdict[p]
				else:
					return p
			
			l_out = ntdict[lhs]
			if isinstance(rhs, tuple):
				r_out = " ".join([print_rhs_part(r) for r in rhs])
			else:
				assert isinstance(rhs, str)
				r_out = print_rhs_part(rhs)
			
			print("{:>8} -> {:15} # {}".format(l_out, r_out, prob))

def intify_prules(grammarbuilder):
	ntdict = {} # index -> NT
	rev_ntdict = {} # NT -> index
	nextindex_d = {'': 1}
	def treat_nt(w):
		if w in rev_ntdict:
			return rev_ntdict[w]
		else:
			index = nextindex_d['']
			ntdict[index] = w
			rev_ntdict[w] = index
			nextindex_d[''] = index + 1
			return index
	
	prules = grammarbuilder(treat_nt)
	
	return prules, ntdict

def normalize_rules(crisp_rules_iter):
	# dict-ize
	crisp_rules = defaultdict(list)
	for (lhs, rhs) in crisp_rules_iter:
		crisp_rules[lhs].append(rhs)
	
	# normalize to get probs
	prules = defaultdict(list)
	for lhs in crisp_rules.keys():
		Z = float(len(crisp_rules[lhs]))
		for rhs, count in Counter(crisp_rules[lhs]).items():
			prules[lhs].append((rhs, float(count) / Z))
	
	return prules

# Transform grammar into CNF (but allowing chain rules) for parsing
def cnfize_grammar(inprules, inntdict):
	ntdict = {k: v for (k,v) in inntdict.items()}
	rev_ntdict = {v: k for (k,v) in inntdict.items()}
	
	nextindex_d = {'': max(ntdict.keys()) + 1}
	
	def binarize(lhs, rhs):
		if len(rhs) == 2:
			r1, r2 = rhs
			return [(lhs, (r1, r2))]
		elif len(rhs) == 1:
			# We'll allow chain rules!
			return [(lhs, (rhs[0],))]
		else:
			# let's do... uh... right-branching!
			r1, r2 = rhs[-2:]
			pt1 = str(ntdict[r1])
			pt2 = str(ntdict[r2])
			pt1 = ("_" if pt1[0] != '_' else "") + pt1
			pt2 = ("_" if pt2[0] != '_' else "") + pt2
			newnt = pt1 + pt2
			if newnt not in rev_ntdict:
				newindex = nextindex_d['']
				nextindex_d[''] = newindex + 1
				ntdict[newindex] = newnt
				rev_ntdict[newnt] = newindex
			else:
				newindex = rev_ntdict[newnt]
			return binarize(lhs, rhs[:-2] + (newindex,)) + [(newindex, (r1, r2))]
	
	outrules = defaultdict(list)
	for (lhs, rhss) in inprules.items():
		for (rhs, prob) in rhss:
			newrules = binarize(lhs, rhs)
			assert newrules[0][0] == lhs
			outrules[lhs].append((newrules[0][1], prob))
			for (l, r) in newrules[1:]:
				outrules[l].append((r, 1.0))
	
	return outrules, ntdict

# Tree :: (str/label, [Tree/child])
def pp_tree(nt_dict, t):
	l, cs = t
	if cs == []:
		return l
	else:
		return nt_dict[l] + "(" + ", ".join(pp_tree(nt_dict, c) for c in cs) + ")"

# Actual CKY parsing
# First the worker
def parse_sent(word_to_preterminal, nt_chains, rhss_to_lhs, s, return_dict):
	print("parsing:", s)
	sent = s.split()
	
	# CKYchart :: {(int,int): {int/nt: (float/prob, Tree/parsetree)}} # 0-based
	ckychart = {}
	
	# Terminal rules
	#print("Width: \n1: ", end = '')
	for (i,w) in enumerate(sent):
		ws = word_to_preterminal[w]
		#print(len(ws), "/ ", end = '')
		ckychart[(i,i)] = ws
	#print("")
	
	# Higher levels
	for width in range(2, len(sent) + 1):
		for i in range(0, len(sent) - width + 1):
			ckychart[(i, i + width - 1)] = {}
			cell = ckychart[(i, i + width - 1)]
			for split in range(i + 1, i + width):
				# Generate new candidates by combining two lower cells
				#print(" ".join(sent[i:split]), "|", " ".join(sent[split:i+width]))
				for (r1, (pr1, pt1)) in ckychart[(i, split - 1)].items():
					for (r2, (pr2, pt2)) in ckychart[(split, i + width - 1)].items():
						for (lhs, prob) in rhss_to_lhs[(r1,r2)]:
							new_prob = pr1 + pr2 + prob
							if lhs not in cell or cell[lhs][0] < new_prob:
								new_ptree = (lhs, [pt1, pt2])
								cell[lhs] = (new_prob, new_ptree)
				# Generate new candidates by applying chain rules
				got_new = True
				while got_new:
					got_new = False
					for (reached_nt, (lower_prob, lower_ptree)) in list(cell.items()):
						for (lhs, cr_prob) in nt_chains[reached_nt].items():
							new_prob = cr_prob + lower_prob
							if lhs not in cell or cell[lhs][0] < new_prob:
								got_new = True
								new_ptree = (lhs, [lower_ptree])
								cell[lhs] = (new_prob, new_ptree)
		#print("{}: ".format(width),end="")
		#print(" / ".join([str(len(ckychart[(i, i + width - 1)])) for i in range(0, len(sent) - width + 1)]))
	
	#print("")
	return_dict[''] = ckychart[(0, len(sent) - 1)]

def cky_parse(cnf_rules, test):
	# We assume rules are unique.
	assert len(set(cnf_rules)) == len(cnf_rules)
	
	# Build rule dicts
	# {str/lower: [(int/upper, (float/prob, Tree/parsetree))]}
	word_to_preterminal = defaultdict(list)
	# {int/lower: [(int/upper, float/prob)]}
	nt_chains = defaultdict(list)
	# {int/rhs1: {int/rhs2: [(int/lhs, float/prob)]}}
	rhss_to_lhs = defaultdict(list)
	
	for (lhs, rhss) in cnf_rules.items():
		for (rhs, prob) in rhss:
			r = (lhs, rhs)
			if len(rhs) == 1:
				(t,) = rhs
				if isinstance(t, str):
					parsetree = (lhs, [(t, [])])
					word_to_preterminal[t].append((lhs, (math.log(prob), parsetree)))
				elif isinstance(t, int):
					nt_chains[t].append((lhs, math.log(prob)))
			elif len(rhs) == 2:
				r1, r2 = rhs
				rhss_to_lhs[(r1, r2)].append((lhs, math.log(prob)))
			else:
				raise Exception("Malformed rule " + str(r))
	
	# Transform from lists into dicts
	# {str/lower: {int/upper: (float/prob, Tree/parsetree)}}
	word_to_preterminal = defaultdict(dict, {k: dict(v) for (k,v) in word_to_preterminal.items()})
	# {int/lower: {int/upper: float/prob}}
	nt_chains = defaultdict(dict, {k: dict(v) for (k,v) in nt_chains.items()})
	
	# Specific test sentence parsing
	result = []
	
	for s in test:
		# if len(s.split()) != 58:
		# 	continue
		
		# manager = multiprocessing.Manager()
		# return_dict = manager.dict()
		# p = multiprocessing.Process(target=parse_sent, args=(word_to_preterminal, nt_chains, rhss_to_lhs, s, return_dict))
		# p.start()
		# p.join()
		
		return_dict = {}
		parse_sent(word_to_preterminal, nt_chains, rhss_to_lhs, s, return_dict)
		
		result.append((s, return_dict['']))
	
	return result

# Timing foo
def get_time():
	return (timeit.default_timer(), resource.getrusage(resource.RUSAGE_SELF)[0])
def diff_time(old):
	walltime = timeit.default_timer() - old[0]
	cputime = resource.getrusage(resource.RUSAGE_SELF)[0] - old[1]
	print("Operation took {:5.2f} (walltime) / {:5.2f} (cputime) seconds.".format(walltime, cputime))
def do_time(f, *p, **pp):
	t = get_time()
	res = f(*p, **pp)
	diff_time(t)
	return res

# Get example data
def bananaset():
	s = """
		S -> NP
		NP -> DET NN
		NP -> DET JJ NN
		DET -> 'the'
		DET -> 'the'
		DET -> 'a'
		DET -> 'an'
		JJ -> 'fresh'
		NN -> 'apple'
		NN -> 'orange'
		"""
	
	def parse_handwritten_grammar(treat_nt):
		def getrule(l):
			[lhs, rhs] = l.split("->")
			lhs = treat_nt(lhs.strip())
			rhs = tuple([w[1:-1] if w[0] == "'" and w[-1] == "'" else treat_nt(w) for w in rhs.strip().split()])
			return (lhs, rhs)
		
		return normalize_rules((getrule(l) for l in s.splitlines() if l.strip() != ""))
	
	rules, ntdict = intify_prules(parse_handwritten_grammar)
	test = ["the fresh apple", "the fresh banana"]
	return (rules, ntdict), test

def ptb_wsj_set():
	trainsize = 3500
	testsize = 300
	
	ts = list(itertools.chain(*(treebank.parsed_sents(fid) for fid in treebank.fileids())))[:trainsize + testsize]
	testsents = [" ".join(s.leaves()) for s in ts[trainsize:]]
	
	# with open("/tmp/trees.json", 'w') as f:
	# 	lins = []
	# 	for t in ts[:trainsize]:
	# 		def lin(t):
	# 			if isinstance(t, str):
	# 				label = t
	# 				cs = []
	# 			else:
	# 				label = t.label()
	# 				cs = t
	# 			
	# 			if '"' in label:
	# 				raise Exception("»\"« occuring in data! :O")
	# 			
	# 			return "{\"label\": \"" + label + "\", \"children\": [" + ",".join([lin(c) for c in cs]) + "]}"
	# 		lins.append(lin(t))
	# 	print("[", ",\n".join(lins), "]", file = f)
	# 
	# with open("/tmp/test.txt", 'w') as f:
	# 	for s in testsents:
	# 		print(s, file = f)
	
	ts = ts[:trainsize]
	
	def parse_treebank_grammar(treat_nt):
		def getrules(t):
			lhs = treat_nt(t.label())
			rhs = []
			for x in t:
				if isinstance(x, str):
					rhs.append(x)
				else:
					rhs.append(treat_nt(x.label()))
					for r in getrules(x):
						yield r
			yield (lhs, tuple(rhs))
		
		return normalize_rules(itertools.chain(*(getrules(t) for t in ts)))
	
	rules, ntdict = intify_prules(parse_treebank_grammar)
	#test = ["the old man", "something was .", "something was at the stock market today .", "Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 ."]
	return (rules, ntdict), testsents

if __name__ == "__main__":
	#print("Reading in grammar from string:")
	#(unbin_rules, unbin_nt), test = do_time(bananaset)
	
	print("Reading in PTB-WSJ sample:")
	(unbin_rules, unbin_nt), test = do_time(ptb_wsj_set)
	
	print_grammar(unbin_rules, unbin_nt)
	
	print("\nBinarizing grammar:")
	cnf_rules, cnf_nt = do_time(cnfize_grammar, unbin_rules, unbin_nt)
	print_grammar(cnf_rules, cnf_nt)
	
	print("\nCKY parsing:")
	parses = do_time(cky_parse, cnf_rules, test)
	for s, cell in parses:
		if cell != {}:
			print("{}".format(s))
			got_s = False
			usablecell = list(filter(lambda t: cnf_nt[t[0]][0] != '_', sorted(cell.items(), key = lambda x: x[1], reverse = True)))
			for (n, (p, ptree)) in usablecell[:10]:
				print("\t{:15} ({:1.20f}) -> {}".format(cnf_nt[n], math.exp(p), pp_tree(cnf_nt, ptree)))
				if cnf_nt[n] == "S":
					got_s = True
			if len(usablecell) > 10:
				print("\t...")
				if not got_s:
					for (n, (p, ptree)) in sorted(cell.items(), key = lambda x: x[1], reverse = True):
						if cnf_nt[n] == "S":
							print("\t{:15} ({:1.20f}) -> {}".format(cnf_nt[n], math.exp(p), pp_tree(cnf_nt, ptree)))
		else:
			print("{}\n".format(s))
