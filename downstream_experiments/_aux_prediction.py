import os
import sys
import json
import pickle
import logging
import Levenshtein
import pandas as pd

pathlib = sys.argv[1]
model_index = sys.argv[2]
task_id = sys.argv[-2] 
task_file = sys.argv[-1]
subset = pickle.load(open(task_file, 'rb'))[task_id]

ctlib = json.load(pathlib)

def _send_query_fast( snippet, ctlib, ctid):
    cutoff = 0.9
    results = []
    ct = ctlib[ctid]
    for k in ct:
        clss = 'exact'
        if( isinstance(ct[k], set) or isinstance(ct[k], list) ):
            for t in ct[k]:
                score = Levenshtein.ratio(snippet, t)
                if(score > cutoff):
                    if(score>0.80 and score<0.90):
                        clss = '-m80'
                    if(score>0.90):
                        clss = '-m90'
                    results.append( { 'hit': t, 'ct_label': k, 'score': f'{score}-{clss}' } )
        else:
            score = Levenshtein.ratio(snippet, ct[k])
            if(score > cutoff):
                if(score>0.80 and score<0.90):
                    clss = '-m80'
                if(score>.90):
                    clss = '-m90'
                results.append( { 'hit': t, 'ct_label': k, 'score': f'{score}-{clss}' } )

    return results

def exec(subset, ctlib, model_index):
    path_partial = os.path.join( os.getcwd(), f'part-task-{task_id}.tsv' )
    
    logging.basicConfig( format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO )
    logger = logging.getLogger( f'Prediction with model {model_index}')
    
    i=0
    lap = 10000
    lines = []
    for el in subset:
        ctid, pmid, test_text, test_label = el
        results = _send_query_fast( test_text, ctlib, ctid)

        for r in results:
            found_ct_text = r['hit']
            found_ct_label = r['ct_label']
            score = r['score']
            line = f"{ctid}\t{pmid}\t{test_label}\t{found_ct_label}\t{test_text}\t{found_ct_text}\t{score}"
            lines.append(line)
            if(  len(lines) %10000 == 0 ):
                with open( path_partial, 'a' ) as g:
                    g.write( ('\n'.join(lines) )+'\n' )
                lines = []
            
        i += 1
        if( i%10000 == 0 ):
            logger.info(f"\t\tEntry {i}/{len(subset)}")

    if( len(lines) > 0 ):
        with open( path_partial, 'a' ) as g:
            g.write( ('\n'.join(lines) )+'\n' )

exec(subset, ctlib, model_index)