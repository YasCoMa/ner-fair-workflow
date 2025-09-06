from strsimpy import *

'''
The following lines must be added in the place of instalation (/aloy/home/ymartins/miniconda3/envs/matchct_env/lib/python3.10/site-packages/strsimpy/) of this package:
from .jaro_winkler import JaroWinkler
from .longest_common_subsequence import LongestCommonSubsequence
from .overlap_coefficient import OverlapCoefficient

So that, it is possible to use the new similarity metrics implemented.
'''

def compute_levenshtein(a, b):
	dist = normalized_levenshtein.NormalizedLevenshtein().distance(a, b)
	return dist

def compute_damerau(a, b):
	dist = damerau.Damerau().distance(a, b)
	return dist

def compute_jaccard(a, b):
	dist = jaccard.Jaccard(1).distance(a, b)
	return dist

def compute_cosine(a, b):
	dist = cosine.Cosine(1).distance(a, b)
	return dist

def compute_jaro_winkler(a, b):
	dist = jaro_winkler.JaroWinkler().distance(a, b)
	return dist

def compute_longest_common_subsequence(a, b):
	dist = longest_common_subsequence.LongestCommonSubsequence().distance(a, b)
	return dist

def compute_metric_lcs(a, b):
	dist = metric_lcs.MetricLCS().distance(a, b)
	return dist

def compute_ngram(a, b):
	dist = ngram.NGram().distance(a, b)
	return dist

def compute_optimal_string_alignment(a, b):
	dist = optimal_string_alignment.OptimalStringAlignment().distance(a, b)
	return dist

def compute_overlap_coefficient(a, b):
	dist = overlap_coefficient.OverlapCoefficient().distance(a, b)
	return dist

def compute_qgram(a, b):
	dist = qgram.QGram().distance(a, b)
	return dist

def compute_sorensen_dice(a, b):
	dist = sorensen_dice.SorensenDice().distance(a, b)
	return dist