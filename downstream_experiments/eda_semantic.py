import os
import re
import sys
import pickle

import rdflib

import pandas as pd
from tqdm import tqdm
import plotly.express as px
from scipy.stats import ranksums
from scipy import stats

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class ExplorationSemanticResults:
    def __init__(self, fout):
        self.out = fout
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
        '''
        Acquire computed data and put them together: 
        cp nerfairwf_experiments/trials/biobert-*-hypersearch-biobert-base-cased-v1.2-finetuned-ner/experiment_metadata/experiment_graph_biobert-*-hypersearch.ttl ./out_eda_semantic/data
        cp nerfairwf_experiments/trials/biobert-*-hypersearch-biobert-base-cased-v1.2-finetuned-ner/experiment_metadata/experiment_graph_biobert-*-hypersearch.ttl ./out_eda_semantic/data

        '''

    def load_graphs(self):
        g = rdflib.Graph()
        indir = os.path.join(self.out, 'data')
        for f in os.listdir(indir):
            path = os.pathjoin(indir, f)
            g.parse(path)

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/out_eda_semantic'
    i = ExplorationSemanticResults( odir )
    i.run()