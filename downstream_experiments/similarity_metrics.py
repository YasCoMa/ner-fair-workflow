import re
import pandas as pd
from strsimpy import *

def __normalize_string( s):
        s = s.lower()
        s = ' '.join( re.findall(r'[a-zA-Z0-9\-]+',s) )
        return s

def _process_pair(a, b, with_norm = 'yes'):
    a = str(a)
    b = str(b)
    if(with_norm == 'yes'):
        a = __normalize_string(a)
        b = __normalize_string(b)
    return a, b

def objective_distance(trial):
    metrics = ['levenshtein', 'damerau', 'jaccard', 'cosine', 'jaro_winkler', 'longest_common_subsequence', 'metric_lcs', 'ngram', 'optimal_string_alignment', 'overlap_coefficient', 'qgram', 'sorensen_dice']
    norm = trial.suggest_categorical("normalization", ['no', 'yes'])
    m = trial.suggest_categorical("metric", metrics)
    
    scores = []
    df = pd.read_csv('/aloy/home/ymartins/match_clinical_trial/valout/fast_gold_results_test_validation.tsv', sep='\t')
    tmp = df[ ['ctid', 'pmid', 'test_text', 'test_label'] ]
    df = df[ ['found_ct_text', 'test_text'] ]
    for i in df.index:
        a, b = _process_pair( df.loc[i, 'found_ct_text'], df.loc[i, 'test_text'], norm )
        try:
            dist = eval(f"compute_distance_{m}")(a, b)
        except:
            dist = 'invalid'
            pass
        scores.append(dist)
    tmp['score'] = scores
    tmp = tmp[ tmp.score != 'invalid' ]

    tmp = tmp.groupby( ['ctid', 'pmid', 'test_text', 'test_label'] ).min().reset_index()
    mean = tmp.score.mean()
    
    return mean

def objective_similarity(trial):
    metrics = ['levenshtein', 'damerau', 'jaccard', 'cosine', 'jaro_winkler', 'longest_common_subsequence', 'metric_lcs', 'ngram', 'optimal_string_alignment', 'overlap_coefficient', 'qgram', 'sorensen_dice']
    norm = trial.suggest_categorical("normalization", ['no', 'yes'])
    m = trial.suggest_categorical("metric", metrics)

    scores = []
    df = pd.read_csv('/aloy/home/ymartins/match_clinical_trial/valout/fast_gold_results_test_validation.tsv', sep='\t')
    tmp = df[ ['ctid', 'pmid', 'test_text', 'test_label'] ]
    df = df[ ['found_ct_text', 'test_text'] ]
    for i in df.index:
        a, b = _process_pair( df.loc[i, 'found_ct_text'], df.loc[i, 'test_text'], norm )
        try:
            dist = eval(f"compute_similarity_{m}")(a, b)
        except:
            dist = 'invalid'
            pass
        scores.append(dist)
    tmp['score'] = scores
    tmp = tmp[ tmp.score != 'invalid' ]

    mean = 0
    if( len(tmp) > 0 ): # Not all the metrics have the similarity function implemented
        tmp = tmp.groupby( ['ctid', 'pmid', 'test_text', 'test_label'] ).max().reset_index()
        mean = tmp.score.mean()
    
    return mean

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

def compute_similarity_jaccard(a, b):
	dist = jaccard.Jaccard(1).similarity(a, b)
	return dist

def compute_similarity_cosine(a, b):
	dist = cosine.Cosine(1).similarity(a, b)
	return dist

def compute_similarity_jaro_winkler(a, b):
	dist = jaro_winkler.JaroWinkler().similarity(a, b)
	return dist

def compute_similarity_overlap_coefficient(a, b):
	dist = overlap_coefficient.OverlapCoefficient().similarity(a, b)
	return dist

def compute_similarity_sorensen_dice(a, b):
	dist = sorensen_dice.SorensenDice().similarity(a, b)
	return dist