from strsimpy import *

def compute_levenshtein(a, b):
	dist = normalized_levenshtein.NormalizedLevenshtein().distance(a, b)
	return dist

def compute_damerau():
	dist = damerau.Damerau().distance(a, b)
	return dist

def compute_jaccard():
	dist = jaccard.Jaccard().distance(a, b)
	return dist

def compute_cosine():
	dist = cosine.Cosine().distance(a, b)
	return dist

def compute_jaro_winkler():
	dist = jaro_winkler.JaroWinkler().distance(a, b)
	return dist

def compute_longest_common_subsequence():
	dist = longest_common_subsequence.LongestCommonSubsequence().distance(a, b)
	return dist

def compute_metric_lcs():
	dist = metric_lcs.MetricLCS().distance(a, b)
	return dist

def compute_ngram():
	dist = ngram.NGram().distance(a, b)
	return dist

def compute_optimal_string_alignment():
	dist = optimal_string_alignment.OptimalStringAlignment().distance(a, b)
	return dist

def compute_overlap_coefficient():
	dist = overlap_coefficient.OverlapCoefficient().distance(a, b)
	return dist

def compute_qgram():
	dist = qgram.QGram().distance(a, b)
	return dist

def compute_sorensen_dice():
	dist = sorensen_dice.SorensenDice().distance(a, b)
	return dist