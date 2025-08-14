import os
import sys
import pandas as pd

class ExplorationPICOAttr:
	def __init__(self, fout):
		self.outPredDir = '/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/biobert-base-cased-v1.2-finetuned-ner/prediction/'
        
        self.out = os.path.join(fout, 'eda_pico')
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
            
        self.out_ct_processed = os.path.join( fout, "processed_cts" )
        if( not os.path.isdir( self.out_ct_processed ) ) :
            os.makedirs( self.out_ct_processed )

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
    	