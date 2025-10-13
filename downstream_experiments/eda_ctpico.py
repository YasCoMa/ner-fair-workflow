import os
import re
import sys
import pickle

import pandas as pd
from tqdm import tqdm
import plotly.express as px
from scipy.stats import ranksums
from scipy import stats

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class ExplorationCTPicoResults:
    def __init__(self, fout):
        self.goldfolder = '/aloy/home/ymartins/match_clinical_trial/out_ss_choice_optimization/'
        self.infolder = '/aloy/home/ymartins/match_clinical_trial/experiments/validation/validation/'
        
        self.out = fout
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )

    def generate_table_pred_grouped_top(self):
        files = { 'all_predicted': 'predss_results_test_validation.tsv', 'grouped_by_max_ss': 'grouped_predss_results_validation.tsv', 'top_ranked': 'top_grouped_predss_results_validation.tsv' }
        
        dt = {}
        lines = []
        header = ['Entity', 'Papers (All predicted)', 'Annotations (All predicted)', 'Papers (Grouped by max. SS)', 'Annotations (Grouped by max. SS)', 'Papers (Top ranked)', 'Annotations (Top ranked)']
        lines.append( '\t'.join(header) )
        for k in files:
            col_ent = 'entity'
            if( k == 'all_predicted'):
                col_ent = 'test_label'

            path = os.path.join( self.infolder, files[k])
            df = pd.read_csv( path, sep='\t' )
            ents = df[col_ent].unique()
            for e in ents:
                papers = len( df[ df[col_ent]==e ].pmid.unique() )
                annotations = len( df[ df[col_ent]==e ] )

                if( not e in dt ):
                    dt[e] = []
                dt[e].append(papers)
                dt[e].append(annotations)

        body = list( map( lambda x: [ str(y) for y in ([x]+dt[x]) ], dt ))
        body = list( map( lambda x: '\t'.join(x), body ))
        lines += body
        opath = os.path.join( self.out, 'pico_general_numbers_filtering_steps.tsv')
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()

    def generate_distribution_ssvalue(self):
        path = os.path.join( self.infolder, 'grouped_predss_results_validation.tsv' )
        df = pd.read_csv( path, sep='\t' )
        entities = df.entity.values.tolist()
        values = df.val.values.tolist()
        n1 = len(df)

        path = os.path.join( self.goldfolder, 'grouped_fast_gold_results_validation.tsv')
        df = pd.read_csv( path, sep='\t')
        entities += df.test_label.values.tolist()
        values += df.val.values.tolist()
        n2 = len(df)

        df = pd.DataFrame()
        df['Dataset'] = (['Augmented'] * n1) + (['Gold DS'] * n2)
        df['Entity'] = entities
        df['SS Value'] = values

        fig = px.box(df, x="Entity", y="SS Value", color='Dataset')
        opath = os.path.join( self.out, 'ssvalue_distribution.png')
        fig.write_image( opath )

    def run(self):
        self.generate_table_pred_grouped_top()
        self.generate_distribution_ssvalue()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/out_eda_pico'
    i = ExplorationCTPicoResults( odir )
    i.run()