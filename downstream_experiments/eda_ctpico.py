import os
import re
import sys
import json
import math
import pickle

import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
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

        self.rawinfolder = '/aloy/home/ymartins/match_clinical_trial/experiments/validation/process_ct/clinical_trials/'

        self.tag_file = '/aloy/home/ymartins/match_clinical_trial/experiments/tags.json'
        self.tags = self.__remove_prefix_tags()
        
        self.out = fout
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )

    def __remove_prefix_tags(self):
        target_tags = json.load( open(self.tag_file,'r') )
        categories = list( map( lambda x: x[2:] if( x[:2].lower() in ['o-', 'b-', 'i-', 'e-', 's-', 'u-', 'l-']) else x, target_tags ) )
        
        aux = []
        for j, c in enumerate(categories):
            if( c not in aux ):
                aux.append(c)
        categories = aux

        return categories

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
        opath = os.path.join(self.out, 'compiled_comparison_gold_aug.json')
        if( not os.path.isfile(opath) ):
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

            aux = {}
            for k in d:
                aux[k] = d[k]
                for e in d[k]:
                    aux[k][e]['papers'] = list( d[k][e]['papers'] )

            json.dump( d, open(opath, 'w') )
        else:
            d = json.load( open(opath, 'r') )

        lines = []
        header = ["Dataset", "Entity", "Metric", "Count", "Count (log10)"]
        lines.append( '\t'.join(header) )
        body = []
        for k in d:
            for e in d[k]:
                np = len( d[k][e]['papers'] )
                na = d[k][e]['annotations']
                body.append( [k, e, 'Papers', str(np), str( math.log10(np) ) ] )
                body.append( [k, e, 'Annotations', str(na), str( math.log10(na) ) ] )

        body = list( map( lambda x: '\t'.join(x), body ))
        lines += body
        opath = os.path.join( self.out, 'table_count_beforeGold_afterAugds.tsv')
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()

        df = pd.read_csv(opath, sep='\t')
        ents = df[ df['Dataset'] == 'augmented' ].Entity.unique()
        for m in df.Metric.unique():
            aux = df[ (df['Metric'] == m) & (df.Entity.isin(ents) ) ]
            fig = px.bar( aux, x="Entity", y="Count (log10)", text="Count", color='Dataset', title = f"{m} enrichment", barmode='group')
            opath = os.path.join( self.out, f'count_{m}_augds_gold.png')
            fig.update_layout( autosize=False, width=1200, height=600)
            fig.write_image( opath )

    def _get_cts_with_result(self, ids):
        d = {}
        opath = os.path.join(self.out, 'cts_with_results.json')
        if( not os.path.isfile(opath) ):
            for f in tqdm( os.listdir(self.rawinfolder) ):
                if( f.startswith('raw') ):
                    path = os.path.join( self.rawinfolder, f )
                    dt = json.load( open( path, 'r' ) )
                    for s in dt:
                        _id = s["protocolSection"]["identificationModule"]["nctId"]
                        if( _id in ids ):
                            d[_id] = 0
                            if( s['hasResults'] ):
                                d[_id] = 1
            json.dump( d, open(opath, 'w') )
        else:
            d = json.load( open(opath, 'r') )

        cnt = sum(d.values())
        return cnt

    def get_statistics_ct_files(self):
        d = {}
        cnt = { k: 0 for k in self.tags }
        procs = os.path.join(self.infolder, 'processed_cts')
        for f in tqdm( os.listdir( procs ) ):
            ctid = f.split('.')[0].split('_')[-1]
            d[ctid] = {}
            path = os.path.join(procs, f)
            dt = json.load( open(path, 'r') )
            for t in self.tags:
                d[ctid][t] = 0
                if( t in dt ):
                    d[ctid][t] = len(dt[t])
                    cnt[t] += len(dt[t])

        lines = []
        header = ["CT"] + self.tags + ['total']
        lines.append( '\t'.join(header) )
        body = []
        for k in d:
            cnts = [ str(x) for x in d[k].values() ]
            total = str( sum( d[k].values() ) )
            body.append( [k] + cnts + [total] )

        body = list( map( lambda x: '\t'.join(x), body ))
        lines += body
        opath = os.path.join( self.out, 'table_count_tags_in processed_cts.tsv')
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()
        
        x = list( filter( lambda el: cnt[el] != 0, cnt.keys() ))
        txt = [ cnt[el] for el in x ]
        y = [ math.log10(el) for el in txt ]
        fig = go.Figure( [ go.Bar( x = x, y = y, text = txt) ] )
        opath = os.path.join( self.out, f'count_tags_in_processed_cts.png')
        fig.update_layout( xaxis_title = 'Entities', yaxis_title = 'Count (log10)' )
        fig.write_image( opath )

        ctids = set(d)
        cnt = self._get_cts_with_result(ctids)
        print('Total with results: ', cnt)

    def run(self):
        #self.generate_table_pred_grouped_top()
        #self.generate_distribution_ssvalue()
        self.comparison_measure_augmentation()
        #self.get_statistics_ct_files()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/out_eda_pico'
    i = ExplorationCTPicoResults( odir )
    i.run()
