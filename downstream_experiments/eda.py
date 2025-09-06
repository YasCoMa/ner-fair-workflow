import os
import re
import sys
import pickle
import optuna
import pandas as pd
import plotly.express as px

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )
from downstream_experiments.similarity_metrics import *

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

        # transform results
        scores_nyes = []
        scores_nno = []
        entities = []
        opath = '/aloy/home/ymartins/match_clinical_trial/valout/fast_gold_results_test_validation.tsv'
        df = pd.read_csv( opath, sep='\t')
        for i in df.index:
            entities.append( df.loc[i, 'test_label'] )

            ay, by = _process_pair( df.loc[i, 'found_ct_text'], df.loc[i, 'test_text'], 'yes' )
            an, bn = _process_pair( df.loc[i, 'found_ct_text'], df.loc[i, 'test_text'], 'no' )
            try:
                sim_nyes = compute_similarity_cosine(ay, by)
                sim_nno = compute_similarity_cosine(an, bn)
            except:
                sim_nyes = 0
                sim_nno = 0
            scores_nyes.append(sim_nyes)
            scores_nno.append(sim_nno)

        df['cosine_score_with_norm'] = scores_nyes
        df['cosine_score_without_norm'] = scores_nno
        df.to_csv( opath, sep='\t', index=None)
        
        opath = '/aloy/home/ymartins/match_clinical_trial/valout/cosine_grouped_fast_gold_results_test_validation.tsv'
        df = df.groupby( ['ctid', 'pmid', 'test_text', 'test_label'] ).max().reset_index()
        df.to_csv( opath, sep='\t', index=None)
        
        df = df[ ['test_label', 'cosine_score_with_norm'] ]
        df.columns = ["Entity", 'Cosine similarity']
        fig = px.box(df, x="Entity", y="Cosine similarity", points="all")
        fig.write_image('valout/cosine_gold_grouped_distribution_scoresim.png')

        subdf = pd.DataFrame()
        subdf["Entity"] = entities * 2
        subdf["Cosine similarity"] = scores_nyes + scores_nno
        subdf["Transformation"] = ['With normalization']*len(scores_nyes) + ['Without normalization']*len(scores_nno)
        fig = px.box( subdf, x="Entity", y="Cosine similarity", color = "Transformation", points="all")
        fig.write_image('valout/cosine_gold_all_distribution_scoresim.png')

    def check_best_string_sim_metric(self):
        studies = { 'similarity': 'maximize', 'distance': 'minimize' }
        for s in studies:
            print("Optimizing for ", s)
            func = eval( f'objective_{s}' )
            direction = studies[s]

            study = optuna.create_study( direction = direction )
            study.optimize( func, n_trials = 50 )
            print('\tBest params:', study.best_trial)

            file = open(f"{self.out}/by_{s}_best_params.pkl", "wb")
            pickle.dump(study.best_trial, file)
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
        #self.check_best_string_sim_metric()
        self.get_coverage_gold_ctapi()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/valout'
    i = ExplorationPICOAttr( odir )
    i.run()