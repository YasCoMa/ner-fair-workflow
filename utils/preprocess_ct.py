import os
import re
import json
import time
from tqdm import tqdm
import requests
import pandas as pd

class InformationExtractionCT:
    def __init__(self, outdir):
        self.stopwords = ["other", "key", "inclusion criteria", "exclusion criteria", "not specified", "see disease characteristics"]
    
        self.out = outdir
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
            
    def _check_stopwords( self, term):
        flag = True
        for st in self.stopwords:
            if( term.lower().find(st) != -1 ):
                return False
        
        return flag

    def _process_study( self, outp, s ):
        obj = {}
        
        mi = s['protocolSection']['identificationModule']
        obj["id"] = mi['nctId']
        obj["name"] = "-"
        if( "officialTitle" in mi ):
            obj["name"] = mi['officialTitle']
        else:
            if( "briefTitle" in mi ):
                obj["name"] = mi['briefTitle']
            
        obj["conditions"] = s['protocolSection']['conditionsModule']['conditions']
        
        obj["interventions"] = []
        if( "armsInterventionsModule" in s['protocolSection'] ):
            if( "interventions" in s['protocolSection']['armsInterventionsModule'] ):
                obj["interventions"] = s['protocolSection']['armsInterventionsModule']['interventions']
        
        obj["outcomes"] = []
        if( "outcomesModule" in s['protocolSection'] ):
            obj["outcomes"] = s['protocolSection']['outcomesModule']
        
        try:
            pmids = set()
            ref = s['protocolSection']['referencesModule']['references']
            for r in ref:
                pmids.add(r['pmid'])
            obj["references"] = list(pmids)
        except:    
            obj["references"] = []
            
        obj["healthyVolunteers"] = "-"
        if( "healthyVolunteers" in s['protocolSection']['eligibilityModule'] ):
            obj["healthyVolunteers"] = s['protocolSection']['eligibilityModule']['healthyVolunteers']
        
        obj["sex"] = "-"
        if( "sex" in s['protocolSection']['eligibilityModule'] ):
            obj["sex"] = s['protocolSection']['eligibilityModule']['sex']
        
        obj["minimumAge"] = "-"
        if( "minimumAge" in s['protocolSection']['eligibilityModule'] ):
            obj["minimumAge"] = s['protocolSection']['eligibilityModule']['minimumAge']
        
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
            
            obj['inclusion'] = inc
            obj['exclusion'] = exc
            
        except:
            print( obj['id'], s['protocolSection']['eligibilityModule'] )
            obj = None
        
        return obj
                  
    def get_clinical_trials(self):
        outp = os.path.join( self.out, 'clinical_trials')
        if( not os.path.isdir( outp ) ) :
            os.makedirs( outp )
        
        omap = os.path.join( self.out, 'mapping_pubmed.tsv' )
        fm = open( omap, 'w' )
        fm.write("ctid\tpmid\n")
        fm.close()
        
        root = "https://clinicaltrials.gov/api/v2/studies?filter.overallStatus=COMPLETED&countTotal=true&pageSize=1000"
        r = requests.get(root)
        dat = r.json()
        tgone = 0
        p = 0
        lobj = []
        total = dat["totalCount"]
        print(total, 'studies')
        ns = dat["nextPageToken"]
        for s in tqdm(dat['studies']):
            obj = self._process_study( outp, s )
            if( obj != None ):
                lobj.append(obj)
                    
                _id = obj["id"]
                refs = obj["references"]
                for r in refs:
                    with open(omap, 'a') as fm:
                        fm.write(f"{_id}\t{r}\n")
        tgone += 1000
        p += 1
        ofile = os.path.join( outp, f"completed_cts_page-{ p }.json" )
        json.dump( lobj, open(ofile, 'w') )
            
        while( ns != None and ns != '' ):
            lobj = []
            r = requests.get( root+'&pageToken='+ns )
            dat = r.json()
            ns = None
            if( "nextPageToken" in dat ):
                ns = dat["nextPageToken"]
                
            for s in tqdm(dat['studies']):
                obj = self._process_study( outp, s )
                if( obj != None ):
                    lobj.append(obj)
                    
                    _id = obj["id"]
                    refs = obj["references"]
                    for r in refs:
                        with open(omap, 'a') as fm:
                            fm.write(f"{_id}\t{r}\n")
                
            tgone += 1000
            p += 1
            ofile = os.path.join( outp, f"completed_cts_page-{ p }.json" )
            json.dump( lobj, open(ofile, 'w') )
            print(ns, tgone, total)
            time.sleep(1)
        
    def get_overall_metrics(self):
        omap = os.path.join( self.out, 'mapping_criteriaItem_ctid.tsv' )
        fm = open( omap, 'w' )
        fm.write("ctid\tcriteria\ttype\n")
        fm.close()
    
        cnt = {  }
        outp = os.path.join( self.out, 'clinical_trials')
        for f in os.listdir(outp):
            if( f.startswith('complete') ):
                path = os.path.join( outp, f )
                dt = json.load( open( path, 'r' ) )
                for s in dt:
                    _id = s["id"]
                    
                    if( 'inclusion' in s ):
                        inc = s['inclusion']
                        for it in inc:
                            it = it.lower()
                            if( not it in cnt):
                                cnt[it] = { 'inclusion': 0, 'exclusion': 0, 'all': 0 }
                            cnt[it]['inclusion'] += 1
                            cnt[it]['all'] += 1
                            
                            with open(omap, 'a') as fm:
                                fm.write(f"{_id}\t{it}\tinclusion\n")
                    else:
                        print(s)
                        
                    if( 'exclusion' in s ):
                        exc = s['exclusion']
                        for it in exc:
                            it = it.lower()
                            if( not it in cnt):
                                cnt[it] = { 'inclusion': 0, 'exclusion': 0, 'all': 0 }
                            cnt[it]['exclusion'] += 1
                            cnt[it]['all'] += 1
                            
                            with open(omap, 'a') as fm:
                                fm.write(f"{_id}\t{it}\texclusion\n")
    
        ofile = os.path.join( self.out, 'overall_item_count.tsv')
        f = open(ofile,"w")
        f.write("criteria_item\tcount_inclusion\tcount_exclusion\tcount_all\n")
        for it in cnt:
            nex = cnt[it]['exclusion']
            nin = cnt[it]['inclusion']
            nall = cnt[it]['all']
            f.write( f"{it}\t{nin}\t{nex}\t{nall}\n" )
        f.close()
        
    def get_missing_cts(self):
        outp = os.path.join( self.out, 'clinical_trials')
        if( not os.path.isdir( outp ) ) :
            os.makedirs( outp )

        ids=set()
        for f in os.listdir('out/clinical_trials/'):
            if( f.startswith('complete') ):
                path = os.path.join( 'out/clinical_trials/', f )
                dt = json.load( open( path, 'r' ) )
                for s in dt:
                    ids.add(s['id'])
        ids = set()
        root = "https://clinicaltrials.gov/api/v2/studies?filter.overallStatus=COMPLETED&countTotal=true&pageSize=1000"
        r = requests.get(root)
        dat = r.json()
        tgone = 0
        p = 0
        lobj = []
        total = dat["totalCount"]
        print(total, 'studies')
        ns = dat["nextPageToken"]
        for s in tqdm(dat['studies']):
            if( s['protocolSection']['identificationModule']['nctId'] not in ids ):
                lobj.append(s)
                
        tgone += 1000
        p += 1
        ofile = os.path.join( outp, f"raw_completed_cts_page-{ p }.json" )
        json.dump( lobj, open(ofile, 'w') )
            
        while( ns != None and ns != '' ):
            lobj = []
            r = requests.get( root+'&pageToken='+ns )
            dat = r.json()
            ns = None
            if( "nextPageToken" in dat ):
                ns = dat["nextPageToken"]
                
            for s in tqdm(dat['studies']):
                if( s['protocolSection']['identificationModule']['nctId'] not in ids ):
                    lobj.append(s)
                
            tgone += 1000
            p += 1
            ofile = os.path.join( outp, f"raw_completed_cts_page-{ p }.json" )
            json.dump( lobj, open(ofile, 'w') )
            print(ns, tgone, total)
            time.sleep(1)
    
    def exploratory_analysis(self):
        without_result=0
        ids=set()
        units = set()
        params = set()
        files = list( filter( lambda x: x.startswith('raw'), os.listdir('out/clinical_trials/') ))
        for f in tqdm( files ) :
            path = os.path.join( 'out/clinical_trials/', f )
            dt = json.load( open( path, 'r' ) )
            for s in tqdm(dt):
                ids.add( s['protocolSection']['identificationModule']['nctId'] )
                try:
                    arr = s['resultsSection']['outcomeMeasuresModule']['outcomeMeasures']
                    for it in arr:
                        units.add( it['unitOfMeasure'] )
                        params.add( it['paramType'] )
                except:
                    without_result+=1
                    pass
                        
        print('total', len(ids))
        print( 'with results', len(ids)-without_results )
        
    def make_mapping_ct_pubmed(self):
        all_refs = set()
        omap = os.path.join( self.out, 'complete_mapping_pubmed.tsv' )
        gone = set()
        if( os.path.isfile(omap) ):
            fm = open( omap, 'r' )
            for line in fm:
                l = line.replace('\n','')
                if( (len(l) > 2) and (not l.startswith('pmid') ) ):
                    gone.add( l.split('\t')[0] )
                    all_refs.add( l.split('\t')[1] )
            fm.close()
        else:
            fm = open( omap, 'w' )
            fm.write("ctid\tpmid\n")
            fm.close()
        
        files = list( filter( lambda x: x.startswith('raw'), os.listdir('out/clinical_trials/') ))
        for f in tqdm( files ) :
            path = os.path.join( 'out/clinical_trials/', f )
            dt = json.load( open( path, 'r' ) )
            for s in dt:
                _id = s['protocolSection']['identificationModule']['nctId']
                if( not _id in gone ):
                    try:
                        pmids = set()
                        ref = s['protocolSection']['referencesModule']['references']
                        for r in ref:
                            try:
                                pmids.add(r['pmid'])
                                all_refs.add(r['pmid'])
                            except:
                                pass
                        for r in pmids:
                            with open(omap, 'a') as fm:
                                fm.write(f"{_id}\t{r}\n")
                    except:
                        pass
                    
        print('Total # of articles', len(all_refs) )

        '''
def find_study(id):
    files = list( filter( lambda x: x.startswith('raw'), os.listdir('out/clinical_trials/') ))
    for f in tqdm( files ) :
        path = os.path.join( 'out/clinical_trials/', f )
        dt = json.load( open( path, 'r' ) )
        for s in dt:
            _id = s['protocolSection']['identificationModule']['nctId']
            if( _id == id ):
                return s
    return None
        '''

    def get_gold_ct_pubmed(self):
        df = pd.read_csv('out/complete_mapping_pubmed.tsv', sep='\t')
        os.system("grep -ic 'nct0' experiments/data/*.txt | grep -v ':0' > gold_pmids")
        ids = list( filter( lambda x: x!='', open('gold_pmids').read().split('\n') ))
        ids = list( map( lambda x: int(x.split('.txt')[0].split('/')[-1]), ids ))
        f = df[ df.pmid.isin(ids) ] # now there are 129 articles out of the 160 that were curated by humans that are mapped into CTs
        
                    
    def check_coverage_picos(self):
        gold = list( filter( lambda x: x.endswith('.ann'), os.listdir('./PICO-Corpus/pico_corpus_brat_annotated_files/') ))
        gold = set( list( map( lambda x: x.split('.')[0], gold )) )
        print('Articles in PICO:', len(gold) )
        
        inmap = os.path.join( self.out, 'mapping_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        ids = set( [ str(v) for v in df['pmid'].values ] )
        print('Articles in CTs:', len(ids) )
        
        inter = gold.intersection( ids )
        print('Incommon:', len(inter) )
        '''
        Articles in PICO: 1011
        Articles in CTs: 87336
        Incommon: 74
        '''
    
    def run(self):
        #self.get_clinical_trials()
        #self.get_overall_metrics()
        #self.check_coverage_picos()
        
        #self.get_missing_cts()
        self.make_mapping_ct_pubmed()
        
if( __name__ == "__main__" ):
    odir = './out'
    i = InformationExtractionCT( odir )
    i.run()
