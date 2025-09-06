from strsimpy import *

'''
The following lines must be added in the place of instalation (/aloy/home/ymartins/miniconda3/envs/matchct_env/lib/python3.10/site-packages/strsimpy/) of this package:
from .jaro_winkler import JaroWinkler
from .longest_common_subsequence import LongestCommonSubsequence
from .overlap_coefficient import OverlapCoefficient

To be added in the strsimpy/__init__.py file

So that, it is possible to use the new similarity metrics implemented.
'''

def compute_distance_levenshtein(a, b):
	dist = normalized_levenshtein.NormalizedLevenshtein().distance(a, b)
	return dist

def compute_distance_damerau(a, b):
	dist = damerau.Damerau().distance(a, b)
	return dist

def compute_distance_jaccard(a, b):
	dist = jaccard.Jaccard(1).distance(a, b)
	return dist

def compute_distance_cosine(a, b):
	dist = cosine.Cosine(1).distance(a, b)
	return dist

def compute_distance_jaro_winkler(a, b):
	dist = jaro_winkler.JaroWinkler().distance(a, b)
	return dist

def compute_distance_longest_common_subsequence(a, b):
	dist = longest_common_subsequence.LongestCommonSubsequence().distance(a, b)
	return dist

def compute_distance_metric_lcs(a, b):
	dist = metric_lcs.MetricLCS().distance(a, b)
	return dist

def compute_distance_ngram(a, b):
	dist = ngram.NGram().distance(a, b)
	return dist

def compute_distance_optimal_string_alignment(a, b):
	dist = optimal_string_alignment.OptimalStringAlignment().distance(a, b)
	return dist

def compute_distance_overlap_coefficient(a, b):
	dist = overlap_coefficient.OverlapCoefficient().distance(a, b)
	return dist

def compute_distance_qgram(a, b):
	dist = qgram.QGram().distance(a, b)
	return dist

def compute_distance_sorensen_dice(a, b):
	dist = sorensen_dice.SorensenDice().distance(a, b)
	return dist

def compute_similarity_levenshtein(a, b):
	dist = normalized_levenshtein.NormalizedLevenshtein().similarity(a, b)
	return dist

def compute_similarity_damerau(a, b):
	dist = damerau.Damerau().similarity(a, b)
	return dist

def compute_similarity_jaccard(a, b):
	dist = jaccard.Jaccard(1).similarity(a, b)
	return dist

def compute_similarity_cosine(a, b):
	dist = cosine.Cosine(1).similarity(a, b)
	return dist

def compute_similarity_jaro_winkler(a, b):
	dist = jaro_winkler.JaroWinkler().similarity(a, b)
	return dist

def compute_similarity_longest_common_subsequence(a, b):
	dist = longest_common_subsequence.LongestCommonSubsequence().similarity(a, b)
	return dist

def compute_similarity_metric_lcs(a, b):
	dist = metric_lcs.MetricLCS().similarity(a, b)
	return dist

def compute_similarity_ngram(a, b):
	dist = ngram.NGram().similarity(a, b)
	return dist

def compute_similarity_optimal_string_alignment(a, b):
	dist = optimal_string_alignment.OptimalStringAlignment().similarity(a, b)
	return dist

def compute_similarity_overlap_coefficient(a, b):
	dist = overlap_coefficient.OverlapCoefficient().similarity(a, b)
	return dist

def compute_similarity_qgram(a, b):
	dist = qgram.QGram().similarity(a, b)
	return dist

def compute_similarity_sorensen_dice(a, b):
	dist = sorensen_dice.SorensenDice().similarity(a, b)
	return dist