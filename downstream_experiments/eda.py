import os
import re
import sys
import pickle
import optuna
import pandas as pd
import plotly.express as px
from scipy.stats import ranksums

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class ExplorationBenchmarkDss:
    def __init__(self, fout):
        self.ingold = '/aloy/home/ymartins/match_clinical_trial/valout/fast_gold_results_test_validation.tsv'
        self.outPredDir = '/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/biobert-base-cased-v1.2-finetuned-ner/prediction/'
        
        self.out = os.path.join(fout, 'eda_pico')
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
            
        self.out_ct_processed = os.path.join( fout, "processed_cts" )

    # -------- General
    def wrap_dss_eval_metrics_reprod(self):
        evaluation_modes = ['seqeval-default', 'seqeval-strict', 'sk-with-prefix', 'sk-without-prefix']
        levels = ['ltoken', 'lword']
        target_metrics = ['mcc', 'f1-score']

        indir = '/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/trials/biobert-__db__-hypersearch-biobert-base-cased-v1.2-finetuned-ner'
        dss = [ "bc5cdr", "ncbi", "biored", "chiads", "merged_train"]
        
        # Choose mode eval with best metric values (f1 and mcc)
        best = {}
        for ds in dss:
            pathdir = os.path.join( indir.replace('__db__', ds), 'test', 'summary_reports' )
            for level in levels:
                key = f"{ds}_{level}"
                if(not key in best):
                    best[key] = {}

                for m in target_metrics:
                    dat = {}
                    for mode in evaluation_modes:
                        fname = f"{mode}_summary-report_test-model-{level}.tsv"
                        path = os.path.join(pathdir, fname)
                        df = pd.read_csv( path, sep= '\t')
                        aux = df[ (df['evaluation_metric'] == m) & (df['stats_agg_name'] == 'mean') ]
                        dat[mode] = aux.stats_agg_value.max()
                    odat = dict( sorted( dat.items(), key=lambda item: item[1], reverse=True ) )
                    best[key][m] = list(odat)[0]


    def __solve_retrieve_processed_cts(self, allids):
        gone = set()
        for _id in allids:
            path = os.path.join( self.out_ct_processed, f"proc_ct_{_id}.json" )
            if( os.path.isfile(path) ):
                gone.add(_id)
        todo = set(allids) - gone
        print('todo', len(todo))
        '''
        cts = self._retrieve_ct_studies(todo)
        for s in tqdm(cts):
            _id = s["protocolSection"]["identificationModule"]["nctId"]
            _ = self._get_ct_info(s)
        '''
        not_found = 0
        dat = {}
        for _id in allids:
            path = os.path.join( self.out_ct_processed, f"proc_ct_{_id}.json" )
            if( os.path.isfile(path) ):
                dat[_id] = json.load( open(path, 'r') )
            else:
                not_found += 1
        print( 'not found', not_found) # 1662

        return dat

    def __aggregate_nctids(self):
        allids = set()
        for f in os.listdir(self.out):
            if( f.startswith('general_mapping_') ):
                sourcect = os.path.join( self.out, f)
                df = pd.read_csv( sourcect, sep='\t' )
                ctids = set(df.ctid.unique())
                allids = allids.union(ctids)
        return allids


    def get_coverage_gold_general(self):
        print("")

    def run(self):
        #self.check_best_string_sim_metric()
        self.get_coverage_gold_ctapi()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/valout'
    i = ExplorationPICOAttr( odir )
    i.run()