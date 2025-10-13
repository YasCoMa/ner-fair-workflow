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
        self.goldraw = "/aloy/home/ymartins/match_clinical_trial/experiments/data/"
        self.goldfolder = '/aloy/home/ymartins/match_clinical_trial/out_ss_choice_optimization/'
        self.augdir = "/aloy/home/ymartins/match_clinical_trial/experiments/validation/augmented_data/"
        
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

    def comparison_measure_augmentation(self):
        data_dirs = { 
        "gold": [ self.goldraw, list( filter( lambda x: x.endswith('.ann'), os.listdir(self.goldraw) )) ], 
        "augmented": [ self.augdir, list( filter( lambda x: x.endswith('.ann'), os.listdir(self.augdir) )) 
        ] }
        d = {}
        for k in data_dirs:
            d[k] = {} 

            pathdir = data_dirs[k][0]
            files = data_dirs[k][1]
            files = list( map( lambda x: os.path.join(pathdir, x) , files ))
            for f in tqdm(files):
                pmid = f.split('/')[-1].split('_')[0]
                g = open(f, 'r')
                for line in g:
                    line = line.split('\t')
                    if( len(line) > 1 ):
                        ent = line[1].split(' ')[0]
                        if( not ent in d[k] ):
                            d[k][ent] = { 'papers': set(), 'annotations': 0 }
                        d[k][ent]['annotations'] += 1
                        d[k][ent]['papers'].add(pmid)
                g.close()

        lines = []
        header = ["Dataset", "Entity", "Metric", "Count"]
        lines.append( '\t'.join(header) )
        body = []
        for k in d:
            for e in d[k]:
                np = len( d[k][e]['papers'] )
                na = d[k][e]['annotations']
                body.append( [k, e, 'Papers', str(np) ] )
                body.append( [k, e, 'Annotations', str(na) ] )

        body = list( map( lambda x: '\t'.join(x), body ))
        lines += body
        opath = os.path.join( self.out, 'table_count_beforeGold_afterAugds.tsv')
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()

        df = pd.read_csv(opath, sep='\t')
        for m in df.Metric.unique():
            aux = df[ df['Metric'] == m ]
            fig = px.bar( aux, x="Entity", y="Count", color='Dataset')
            opath = os.path.join( self.out, f'count_{m}_augds_gold.png')
            fig.write_image( opath )

    def run(self):
        #self.generate_table_pred_grouped_top()
        #self.generate_distribution_ssvalue()
        self.comparison_measure_augmentation()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/out_eda_pico'
    i = ExplorationCTPicoResults( odir )
    i.run()