import os
import re
import json
import faiss
import pandas as pd
from tqdm import tqdm
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

'''
# https://python.langchain.com/docs/integrations/vectorstores/faiss/#saving-and-loading

Experiment on gold dataset of annotated PICO entities

1- Check the CT ids mentioned in the abstract documents
2- Get the snippets and their labels
3- Generate embedding faiss index ctinfo
4- same as above but index snippetinfo
'''

class ExperimentValidationBySimilarity:
    def __init__(self, fout):
        self.ctDir = '/aloy/home/ymartins/match_clinical_trial/out/clinical_trials/'
        self.goldDir = '/aloy/home/ymartins/match_clinical_trial/experiments/data/'
        self.inPredDir = '/aloy/home/ymartins/match_clinical_trial/experiments/new_data/'
        
        
        self.out = fout
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
            
        self.out_ct_processed = os.path.join( self.out, "processed_cts" )
        if( not os.path.isdir( self.out_ct_processed ) ) :
            os.makedirs( self.out_ct_processed )
            
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.gold_ct_index = index = faiss.IndexFlatL2( len( self.embeddings.embed_query("hello world") ) )
        self.gct_vs = FAISS( embedding_function = self.embeddings, index = self.gold_ct_index, docstore = InMemoryDocstore(), index_to_docstore_id = {}, ) # original CT info
        self.gold_ann_index = index = faiss.IndexFlatL2( len( self.embeddings.embed_query("hello world") ) )
        self.gann_vs = FAISS( embedding_function = self.embeddings, index = self.gold_ann_index, docstore = InMemoryDocstore(), index_to_docstore_id = {}, ) #  Snippets labeled
        
    def _get_snippets_labels(self, pmid):
        anns = []
        f = f"{pmid}.ann"
        path = os.path.join( self.goldDir, f)
        f = open(path, 'r')
        for line in f:
            l = line.replace('\n','').split('\t')
            label = l[1].split(' ')[0]
            term = l[2]
            anns.append( [term, label] )
        f.close()
        
        return anns
    
    def _map_nctid_pmid_gold(self):
        omap = os.path.join( self.out, 'goldds_labelled_mapping_nct_pubmed.tsv')
        f = open( omap, 'w' )
        f.write( 'pmid\tctid\ttext\tlabel\n' )
        f.close()
        
        for f in os.listdir( self.goldDir ):
            if( f.endswith('.txt') ):
                pmid = f.split('.')[0]
                
                path = os.path.join( self.goldDir, f)
                abs = open(path).read()
                tokens = abs.split(' ')
                ncts = list( filter( lambda x: (x.find('NCT0') != -1), tokens ))
                if( len(ncts) > 0 ):
                    for nid in ncts:
                        tr = re.findall( r'(NCT[0-9]+)', nid )
                        if( len(tr) > 0 ):
                            ctid = tr[0]
                            anns = self._get_snippets_labels( pmid )
                            for a in anns:
                                line = '\t'.join( [pmid, ctid]+a )
                                with open( omap, 'a' ) as g:
                                    g.write( line+'\n' )
    
    def _get_snippets_pred_labels(self, pmid):
        f = self.predictions_df[ self.predictions_df['input_file'].str.startswith(pmid) ]
        anns = []
        for i in f.index:
            label = self.predictions_df.loc[i, 'entity_group']
            term = self.predictions_df.loc[i, 'word']
            anns.append( [term, label] )
        
        return anns
    
    def __load_mapping_pmid_nctid(self):
        mapp = {}
        opath = os.path.join( self.out, 'mapping_ct_pubmed.json' )
        if( not os.path.isfile(opath) ):
            for f in tqdm( os.listdir( self.inPredDir ) ):
                if( f.endswith('.txt') ):
                    pmid = f.split('_')[0]
                    if( pmid not in mapp ):
                        mapp[pmid] = set()
                    path = os.path.join( self.inPredDir, f)
                    abs = open(path).read()
                    tokens = abs.split(' ')
                    ncts = list( filter( lambda x: (x.find('NCT0') != -1), tokens ))
                    if( len(ncts) > 0 ):
                        for nid in ncts:
                            tr = re.findall( r'(NCT[0-9]+)', nid )
                            if( len(tr) > 0 ):
                                for t in tr:
                                    if( t.startswith('NCT') ):
                                        ctid = t
                                        mapp[pmid].add(ctid)
            for k in mapp:
                mapp[k] = list(mapp[k])
            json.dump( mapp, open(opath, 'w') )
        else:
            mapp = json.load( open(opath, 'r') )
            for k in mapp:
                mapp[k] = set(mapp[k])
        return mapp

    def _map_nctid_pmid_general(self, label_exp):
        mapp = self.__load_mapping_pmid_nctid()

        for f in os.listdir( self.outPredDir ):
            if( f.startswith('results_') ):
                fname = f.split('.')[0].replace('results_','')
                omap = os.path.join( self.out, f'general_mapping_{label_exp}_{fname}_nct_pubmed.tsv')
                g = open( omap, 'w' )
                g.write( 'pmid\tctid\ttext\tlabel\n' )
                g.close()
                
                path = os.path.join( self.outPredDir, f)
                print('---- in ', path)
                self.predictions_df = pd.read_csv(path, sep='\t')
                for pmid in mapp:
                    cts = mapp[pmid]
                    for ctid in cts:
                        anns = self._get_snippets_pred_labels( pmid )
                        for a in anns:
                            line = '\t'.join( [pmid, ctid]+a )
                            with open( omap, 'a' ) as g:
                                g.write( line+'\n' )
    
    def _retrieve_ct_studies(self, ids):
        studies = []
        for f in tqdm( os.listdir(self.ctDir) ):
            if( f.startswith('raw') ):
                path = os.path.join( self.ctDir, f )
                dt = json.load( open( path, 'r' ) )
                for s in dt:
                    _id = s["protocolSection"]["identificationModule"]["nctId"]
                    if( _id in ids ):
                        studies.append(s)
        return studies
        
    def _treat_eligibility(self, s):
        inc = []
        exc = []
        sex = set()
        age = set()
        try:
            e = s['protocolSection']['eligibilityModule']['eligibilityCriteria']
            
            parts = e.split('Exclusion Criteria:')
            inc = []
            conds = list( filter( lambda x: x!= "", parts[0].split('\n') ))
            conds = list( map( lambda x: x.replace('\\>', '>').replace('\\<', '<').replace('\\^', '^').replace('≤', '<=').replace('≥', '>=').replace('\\[', '[').replace('\\]', ']'), conds ))
            conds = list( map( lambda x: re.sub('(\*+) |([0-9]+)\. ','', x ).strip(), conds ))
            conds = list( filter( lambda x: ( not x.endswith(':') and self._check_stopwords(x) ), conds ))
            inc = conds
            
            exc = []
            if( len(parts) > 1 ):
                conds = list( filter( lambda x: x!= "", parts[1].split('\n') ))
                conds = list( map( lambda x: x.replace('\\>', '>').replace('\\<', '<').replace('\\^', '^').replace('≤', '<=').replace('≥', '>=').replace('\\[', '[').replace('\\]', ']'), conds ))
                conds = list( map( lambda x: re.sub('(\*+) |([0-9]+)\. ','', x ).strip(), conds ))
                conds = list( filter( lambda x: ( not x.endswith(':') and self._check_stopwords(x) ), conds ))
                exc = conds
            
        except:
            pass
            
        try:
            keys = ['minimumAge', 'maximumAge']
            for k in keys:
                if(k in s['protocolSection']['eligibilityModule']):
                    age.add( s['protocolSection']['eligibilityModule'][k])
            if( 'sex' in s['protocolSection']['eligibilityModule']):
                gender = s['protocolSection']['eligibilityModule']['sex']
                if( gender.lower() == 'all' ):
                    sex.add( 'male' )
                    sex.add( 'female' )
                    sex.add('ALL')
                else:
                    sex.add( gender )
            
        except:
            pass
        
        return [inc, exc, sex, age]
    
    def _get_ct_info(self, s ):
        _id = s["protocolSection"]["identificationModule"]["nctId"]
        path = os.path.join( self.out_ct_processed, f"proc_ct_{_id}.json" )
        if( not os.path.isfile(path) ):
            clabels = set()
            ilabels = set()
            cids = set()
            itids = set()
            control_groups = set()
            interv_groups = set()
            
            condition = set()
            out_measures = set()
            eligibility = set()
            outcomes = set()
            location = set()
            age = set()
            gender = set()
            ethnicity = set()
            
            totp = -1
            controlp = -1
            intervp = -1
            
            # For P entities
            try:
                mintervs = s['protocolSection']['armsInterventionsModule']
                groups = mintervs['armGroups']
                clabels = set()
                ilabels = set()
                for g in groups:
                    if( g['type'] == 'NO_INTERVENTION' ):
                        clabels.add( g['label'] )
                        control_groups.add( g['label'] )
                    else:
                        ilabels.add( g['label'] )
            except:
                pass
            
            params = self._treat_eligibility(s)
            eligibility.update( params[0]+params[1] )
            inclusion = params[0]
            exclusion = params[1]
            gender.update( list(params[2]) )
            age.update( list(params[3]) )
            
            try:
                mconds = s['protocolSection']['conditionsModule']
                condition.update( mconds['conditions'] )
            except:
                pass
            
            try:
                locs = s['protocolSection']['contactsLocationsModule']['locations']
                for l in locs:
                    location.update( [ l[k] for k in ['facility', 'city', 'country'] ] )
            except:
                pass
            
            try:
                moutcomes = s['protocolSection']['outcomesModule']
                for k in moutcomes:
                    outs = moutcomes[k]
                    for g in outs:
                        outcomes.add( g['measure'] )
            except:
                pass
            
            try:
                moutcomes = s['resultsSection']['outcomeMeasuresModule']['outcomeMeasuresModule']['outcomeMeasures']
                for g in moutcomes:
                    out_measures.add( g['title'] )
            except:
                pass
            
            try:
                mintervs = s['protocolSection']['armsInterventionsModule']
                interventions = mintervs['interventions']
                for g in interventions:
                      interv_groups.add( g['name'] )
                      interv_groups.update( g['otherNames'] )
            except:
                pass
                
            try:
                gresults = s['resultsSection']['baselineCharacteristicsModule']['measures']
                for g in gresults:
                    if( g['title'].find('ethnicity') != -1 ):
                        ethnicity.update( list(map( lambda x: x['title'], g['classes'] )) )
            except:
                pass
            
            # To classify the results numbers in iv or cv
            totid = -1  
            try:
                gresults = s['resultsSection']['baselineCharacteristicsModule']['groups']
                totid = gresults[0]['id']
                for g in gresults:
                    name = g['title']
                    _id = g['id']
                    if( name in clabels ):
                        control_groups.add(name)
                        cids.add(_id)
                    else:
                        if( name.lower() != 'total'):
                            interv_groups.add(name)
                            itids.add(_id)
                        else:
                            totid = _id
            except:
                pass   
            
            try:
                gresults = s['resultsSection']['baselineCharacteristicsModule']['denoms']
                for g in gresults:
                    gid = g['groupId']
                    if( gid in clabels ):
                        controlp += int(g['value'])
                        
                    if( gid == totid ):
                        totp = int(g['value'])
                intervp = totp - controlp
            except:
                pass     
            
            # abs or percent values => bin
            # mean, median, sd, q1, q3 => cont  
            md = {}
            try:
                arr = s['resultsSection']['outcomeMeasuresModule']['outcomeMeasures']
                for it in arr:
                    ntype = 'bin'
                    spec = 'abs'
                    if( it['unitOfMeasure'].lower().find('number')==-1 and it['unitOfMeasure'].lower().find('percentage')==-1 and it['unitOfMeasure'].lower().find('count')==-1 ):
                        ntype = 'cont'
                        spec = it['paramType'].lower()
                    else:
                        if( it['unitOfMeasure'].lower().find('percentage')==-1 ):
                            spec = 'percent'
                        
                    ms = it['classes'][0]['categories'][0]['measurements']
                    for m in ms:
                        val = m['value']
                        gr = 'iv'
                        if( m['groupId'] in cids ):
                            gr='cv'
                        key = f'{gr}-{ntype}-{spec}'
                        if( not key in md):
                            md[key]=set()
                        md[key].add(val) 
            except:
                pass

            # Participants
            if( totp != -1 ):
                md['total-participants'] = totp
            if( intervp != -1 ):
                md['intervention-participants'] = intervp
            if( controlp != -1 ):
                md['control-participants'] = controlp
            if( len(age) > 0 ):
                md['age'] = age
            if( len(eligibility) > 0 ):
                md['eligibility'] = eligibility
            if( len(ethnicity) > 0 ):
                md['ethnicity'] = ethnicity
            if( len(condition) > 0 ):
                md['condition'] = condition
            if( len(location) > 0 ):
                md['location'] = location
            
            # Intervention & Control
            if( len(control_groups) > 0 ):
                md['control'] = control_groups
            if( len(interv_groups) > 0 ):
                md['intervention'] = interv_groups
            
            # Outcome
            if( len(outcomes) > 0 ):
                md['outcome'] = outcomes
            if( len(out_measures) > 0 ):
                md['outcome-Measure'] = out_measures
        
            aux = {}
            for k in md:
                if( isinstance( md[k], set ) ):
                    aux[k] = list(md[k])
                else:
                    aux[k] = md[k]
            json.dump( aux, open(path, 'w') )
        else:
            md = json.load( open(path, 'r') )
        
        return md
    
    def embed_save_pubmed(self):
        ctids = set()
        docs = []
        omap = os.path.join( self.out, 'goldds_labelled_mapping_nct_pubmed.tsv')
        df = pd.read_csv( omap, sep='\t' )
        for i in df.index:
            ctid = df.loc[i, 'ctid']
            ctids.add(ctid)
            
            text = df.loc[i, 'text']
            pmid = df,loc[i, 'pmid']
            label = df.loc[i, 'label']
            
            doc = Document( page_content = text, metadata = { "source": str(pmid), "ctid": ctid, "label": label } )
            docs.append(doc)
            
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.gann_vs.add_documents(documents=documents, ids=uuids)
    
    def _extract_info_ct(self):
        mapp = self.__load_mapping_pmid_nctid()
        print('Articles', len(mapp)) # 68744
        ctids = set()
        for v in mapp.values():
            ctids = ctids.union(v)
        print('CTs', len(ctids) ) # 51769
        cts = self._retrieve_ct_studies(ctids) # available 50068
        for s in tqdm(cts): 
            _ = self._get_ct_info(s)

    def embed_save_ncict(self, sourcect, label_ct_index='ctdoc_'):
        path = os.path.join(self.out, f'{label_ct_index}_faiss.index')
        if( os.path.exists(path) ):
            self.gct_vs = FAISS.load_local( path, self.embeddings, allow_dangerous_deserialization=True )
        else:
            ctids = set()
            docs = []
            df = pd.read_csv( sourcect, sep='\t' )
            ctids = set(df.ctid.unique())
            cts = self._retrieve_ct_studies(ctids)
            for s in cts:
                _id = s["protocolSection"]["identificationModule"]["nctId"]
                snippets = self._get_ct_info(s)
                for label in snippets:
                    if( label not in ['inclusion', 'exclusion'] ):
                        text = snippets[label]
                        if( isinstance(text, set) ):
                            for t in text:
                                doc = Document( page_content = t, metadata = { "source": str(_id), "ctid": _id, "label": label } )
                                docs.append(doc)
                        else:
                            doc = Document( page_content = str(text), metadata = { "source": str(_id), "ctid": _id, "label": label } )
                            docs.append(doc)
                
            uuids = [ str(uuid4()) for _ in range( len(docs) ) ]
            self.gct_vs.add_documents(documents=docs, ids=uuids)
            self.gct_vs.save_local(path)
    
    def _send_query(self, snippet, ctid):
        results = []
        rs = self.gct_vs.similarity_search_with_score( snippet, k = 1, filter = {"source": ctid } )
        for res, score in rs:
            score = float(1 - score) # score is actually distance, the higher it is, less it is the match
            hit = res.page_content
            label = res.metadata['label']
            results.append( { 'hit': hit, 'ct_label': label, 'score': score } )
            #print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
        return results

    def _get_predictions(self, sourcect, label_result=''):
        res = os.path.join( self.out, f'{label_result}_results_test_validation.tsv')
        f = open(res, 'w')
        f.write("ctid\tpmid\ttest_label\tfound_ct_label\ttest_text\tfound_ct_text\tscore\n")
        f.close()

        df = pd.read_csv( sourcect, sep='\t' )
        for i in df.index:
            ctid = df.loc[i, 'ctid']
            pmid = df.loc[i, 'pmid']
            test_text = df.loc[i, 'text']
            test_label = df.loc[i, 'label']
            results = self._send_query(human_text, ctid)
            for r in results:
                found_ct_text = r['hit']
                found_ct_label = r['ct_label']
                score = r['score']
                with open(res, 'a') as g:
                    g.write( f"{ctid}\t{pmid}\t{test_label}\t{found_ct_label}\t{test_text}\t{found_ct_text}\t{score}\n")

    def perform_validation_gold(self):
        self._map_nctid_pmid_gold()
        sourcect = os.path.join( self.out, 'goldds_labelled_mapping_nct_pubmed.tsv')
        self.embed_save_ncict(sourcect, 'ctdoc_')
        self._get_predictions(sourcect, 'gold')
    
    def perform_validation_biobert_allct(self):
        self.outPredDir = '/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/biobert-base-cased-v1.2-finetuned-ner/prediction/'
        self._map_nctid_pmid_general('biobert')
        for f in os.listdir(self.out):
            if( f.startswith('general_mapping_') ):
                fname = f.split('.')[0].replace('general_mapping_','')
                sourcect = os.path.join( self.out, f)
                self.embed_save_ncict(sourcect, f'ctall_biobert_{fname}_')
                self._get_predictions(sourcect, f'general_biobert_{fname}_' )
    
    def perform_validation_longformer_allct(self):
        self.outPredDir = '/aloy/home/ymartins/match_clinical_trial/experiments/longformer_trial/longformer-base-4096-finetuned-ner/prediction/'
        self._map_nctid_pmid_general('longformer')
        for f in os.listdir(self.out):
            if( f.startswith('general_mapping_') ):
                fname = f.split('.')[0].replace('general_mapping_','')
                sourcect = os.path.join( self.out, f)
                self.embed_save_ncict(sourcect, f'ctall_longformer_{fname}_')
                self._get_predictions(sourcect, f'general_longformer_{fname}_' )
    
    def run(self):
        #self.perform_validation_gold()
        #self._extract_info_ct()
        #self.perform_validation_allct()
        self.perform_validation_biobert_allct()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/valout'
    i = ExperimentValidationBySimilarity( odir )
    i.run()
                        
