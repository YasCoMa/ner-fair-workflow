import os
import sys
import optuna
import pandas as pd
import plotly.express as px

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )
from downstream_experiments.similarity_metrics import *

def objective(trial):
    metrics = ['levenshtein', 'damerau', 'jaccard', 'cosine', 'jaro_winkler', 'longest_common_subsequence', 'metric_lcs', 'ngram', 'optimal_string_alignment', 'overlap_coefficient', 'qgram', 'sorensen_dice']
    m = trial.suggest_categorical("metric", metrics)

    scores = []
    df = pd.read_csv('valout/fast_gold_results_test_validation.tsv', sep='\t')
    tmp = df[ ['ctid', 'pmid', 'test_text', 'test_label'] ]
    df = df[ ['found_ct_text', 'test_text'] ]
    for i in df.index:
        a = df.loc[i, 'found_ct_text']
        b = df.loc[i, 'test_text']
        score.append( eval(f"compute_{m}")(a, b) )
    tmp['score'] = score

    tmp = tmp.groupby( ['ctid', 'pmid', 'test_text', 'test_label'] ).max().reset_index()
    score = tmp.score.mean()
    
    return mean

class ExplorationPICOAttr:
	def __init__(self, fout):
		self.outPredDir = '/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/biobert-base-cased-v1.2-finetuned-ner/prediction/'
        
        self.out = os.path.join(fout, 'eda_pico')
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
            
        self.out_ct_processed = os.path.join( fout, "processed_cts" )

    # -------- Gold ds
    def get_coverage_gold_ctapi(self):
        path = '/aloy/home/ymartins/match_clinical_trial/valout/grouped_fast_gold_results_test_validation.tsv'
        df = pd.read_csv( path, sep='\t')
        print("Number of CTs:", len(df.ctid.unique()) ) # 117
        print("Number of PMIDs:", len(df.pmid.unique()) ) # 129

        df = pd.read_csv('/aloy/home/ymartins/match_clinical_trial/valout/grouped_fast_gold_results_test_validation.tsv', sep='\t')
        fig = px.box(df, x="test_label", y="val", points="all")
        fig.write_image('valout/gold_grouped_distribution_scoresim.png')

    def check_best_string_sim_metric(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        print(study.best_trial)

        file = open(f"{self.out}/best_params.pkl", "wb")
        pickle.dump(study.best_params, file)
        file.close()

    # -------- General
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
        self.check_best_string_sim_metric()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/valout'
    i = ExplorationPICOAttr( odir )
    i.run()