import os
import re
import sys
import json
import pickle
import logging
import Levenshtein
import pandas as pd

pathlib = sys.argv[1]
model_index = sys.argv[2]
mode = sys.argv[3]
path_faiss = sys.argv[4]

task_id = sys.argv[-2] 
task_file = sys.argv[-1]
subset = pickle.load(open(task_file, 'rb'))[task_id]

ctlib = json.load( open(pathlib, 'r') )

gct_vs = None
if( mode=='ollama' and path_faiss!='none' and os.path.exists(path_faiss)):
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    gct_vs = FAISS.load_local( path_faiss, embeddings, allow_dangerous_deserialization=True )


def normalize_string(s):
    s = s.lower()
    s = ' '.join( re.findall(r'[a-zA-Z0-9\-]+',s) )
    return s

def _send_query( snippet, ctid, gct_vs):
    results = []
    rs = gct_vs.similarity_search_with_score( snippet, k = 1, filter = {"source": ctid } )
    for res, score in rs:
        score = float(1 - score) # score is actually distance, the higher it is, less it is the match
        hit = res.page_content
        label = res.metadata['label']

        clss = 'exact'
        if( score < 1):
            clss = 'm'+str(score).split('.')[1][0]+'0'
        results.append( { 'hit': hit, 'ct_label': label, 'score': f'{score}-{clss}' } )

    return results

def _send_query_fast( snippet, ctlib, ctid, label='all'):
    cutoff = 0.3
    results = []
    ct = ctlib[ctid]

    keys = list(ct)
    if(label != 'all'):
        tags = []
        if(label in keys):
            tags = [label]
    else:
        tags = keys

    for k in tags:
        try:
            elements = [ ct[k] ]
            if( isinstance(ct[k], set) or isinstance(ct[k], list) ):
                elements = ct[k]
                
            for el in elements:
                el = str(el)
                clss = 'exact'
                
                nel = normalize_string(el)
                nsnippet = normalize_string(snippet)
                
                score = Levenshtein.ratio( nsnippet, nel )
                if(score >= cutoff):
                    if( score < 1):
                        clss = 'm'+str(score).split('.')[1][0]+'0'
                    results.append( { 'hit': el, 'ct_label': k, 'score': f'{score}-{clss}' } )
        except:
            pass
    return results

def exec(subset, ctlib, model_index, mode, gct_vs):
    cts_available = set(ctlib)

    path_partial = os.path.join( os.getcwd(), f'part-task-{task_id}.tsv' )
    
    logging.basicConfig( format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO )
    logger = logging.getLogger( f'Prediction with model {model_index}')
    
    i=0
    lap = 1000
    lines = []
    for el in subset:
        ctid, pmid, test_text, test_label = el
        if( ctid in cts_available ):
            if(mode=='fast' or gct_vs==None):
                results = _send_query_fast( test_text, ctlib, ctid, label=test_label )
            elif(mode=='ollama'):
                results = _send_query( test_text, ctid, gct_vs)
            
            for r in results:
                found_ct_text = r['hit']
                found_ct_label = r['ct_label']
                score = r['score']
                line = f"{ctid}\t{pmid}\t{test_label}\t{found_ct_label}\t{test_text}\t{found_ct_text}\t{score}"
                lines.append(line)
                if(  len(lines) % lap == 0 ):
                    with open( path_partial, 'a' ) as g:
                        g.write( ('\n'.join(lines) )+'\n' )
                    lines = []
            
        i += 1
        if( i % lap == 0 ):
            logger.info(f"\t\tEntry {i}/{len(subset)}")

    if( len(lines) > 0 ):
        with open( path_partial, 'a' ) as g:
            g.write( ('\n'.join(lines) )+'\n' )

exec(subset, ctlib, model_index, mode, gct_vs)