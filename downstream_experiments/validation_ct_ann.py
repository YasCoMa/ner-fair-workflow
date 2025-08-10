import os
import re
import json
import faiss
import pandas as pd
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
        
        self.out = fout
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
            
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
            anns.append( [label, term] )
        f.close()
        
        return anns
    
    def _map_nctid_pmid(self):
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
                        tr = re.findall( r'([a-zA-Z0-9]+)', nid )
                        if( len(tr) > 0 ):
                            ctid = tr[0]
                            anns = self._get_snippets_labels( pmid )
                            for a in anns:
                                line = '\t'.join( [pmid, ctid]+a )
                                with open( omap, 'a' ) as g:
                                    g.write( line+'\n' )
    
    def _retrieve_ct_studies(self, ids):
        studies = []
        for f in os.listdir(self.ctDir):
            if( f.startswith('raw') ):
                path = os.path.join( self.ctDir, f )
                dt = json.load( open( path, 'r' ) )
                for s in dt:
                    if( s['id'] in ids ):
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
            
        criteria = inc + exc
        
        return [criteria, sex, age]
    
    def _get_ct_info(self, s ):
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
        eligibility.update( params[0] )
        gender.update( list(parms[1]) )
        age.update( list(parms[2]) )
        
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
                
            ms = it['classes'][0][categories][0]['measurements']
            for m in ms:
                val = m['value']
                gr = 'iv'
                if( m['groupId'] in cids ):
                    gr='cv'
                key = f'{gr}-{ntype}-{spec}'
                if( not key in md):
                    md[key]=set()
                md[key].add(val) 

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
    
    def embed_save_ncict(self):
        ctids = set()
        docs = []
        omap = os.path.join( self.out, 'goldds_labelled_mapping_nct_pubmed.tsv')
        df = pd.read_csv( omap, sep='\t' )
        ctids = set(df.ctid.unique())
        cts = self._retrieve_ct_studies(ctids)
        for s in cts:
            _id = s["protocolSection"]["identificationModule"]["nctId"]
            snippets = self._get_ct_info(s)
            for label in snippets:
                text = snippets[label]
                if( isinstance(text, set) ):
                    for t in text:
                        doc = Document( page_content = t, metadata = { "source": str(_id), "ctid": _id, "label": label } )
                        docs.append(doc)
                else:
                    doc = Document( page_content = str(text), metadata = { "source": str(_id), "ctid": _id, "label": label } )
                    docs.append(doc)
            
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.gct_vs.add_documents(documents=documents, ids=uuids)
    
    def _send_query(self, snippet, ctid):
        results = self.gct_vs.similarity_search_with_score( snippet, k = 1, filter = {"source": ctid } )
        for res, score in results:
            hit = res.page_content
            label = res.metadata.label
            results.append( { 'hit': hit, 'ct_label': label, 'score': score } )
            #print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
        return results

    def perform_validation_gold(self):
        res = os.path.join( self.out, 'gold_results_test_validation.tsv')
        f = open(res, 'w')
        f.write("ctid\tpmid\thuman_label\tfound_label\thuman_text\tfound_text\tscore\n")
        f.close()

        omap = os.path.join( self.out, 'goldds_labelled_mapping_nct_pubmed.tsv')
        df = pd.read_csv( omap, sep='\t' )
        for i in df.index:
            ctid = df.loc[i, 'ctid']
            pmid = df,loc[i, 'pmid']
            human_text = df.loc[i, 'text']
            human_label = df.loc[i, 'label']
            results = self._send_query(text, ctid)
            for r in results:
                found_text = r['hit']
                found_label = r['ct_label']
                score = r['score']
                with open(res, 'a') as g:
                    f.write( f"{ctid}\t{pmid}\t{human_label}\t{found_label}\t{human_text}\t{found_text}\t{score}\n")

    def run(self):
        self._map_nctid_pmid()
        self.embed_save_ncict()
        self.perform_validation_gold()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/valout'
    i = ExperimentValidationBySimilarity( odir )
    i.run()
                        
