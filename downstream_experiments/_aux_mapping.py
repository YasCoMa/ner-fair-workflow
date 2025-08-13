import os
import sys
import pickle
import logging
import pandas as pd

task_id = sys.argv[1] 
task_file = sys.argv[2]
subset = pickle.load(open(task_file, 'rb'))[task_id]

def _get_snippets_pred_labels(df, pmid):
    f = df[ df['input_file'].str.startswith(pmid) ]
    anns = []
    for i in f.index:
        label = str(df.loc[i, 'entity_group'])
        term = str(df.loc[i, 'word'])
        anns.append( [term, label] )
    
    return anns

def exec(subset):
    path_partial = os.path.join( os.getcwd(), f'part-task-{task_id}.tsv' )
    pathdf = subset[0][-1]
    fname = pathdf.split(os.path.sep)[-1].split('.')[0].replace('results_','')
    df = pd.read_csv(pathdf, sep='\t')
    
    logging.basicConfig( format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO )
    logger = logging.getLogger( f'Validation with model {fname}')

    pathdf = subset[0][-1]
    df = pd.read_csv(pathdf, sep='\t')
    
    i=0
    lines = []
    for el in subset:
        pmid, mapp, path_df = el

        anns = _get_snippets_pred_labels( df, pmid )
        cts = mapp[pmid]
        for ctid in cts:
            if( len(anns) > 0 ):
                for a in anns:
                    items = [str(pmid), ctid]+a
                    if( len(items) == 4 ):
                        line = '\t'.join( items )
                        lines.append(line)
                        if(  len(lines) %1000 == 0 ):
                            with open( path_partial, 'a' ) as g:
                                g.write( ('\n'.join(lines) )+'\n' )
                            lines = []
            
        i += 1
        if( i%100 == 0 ):
            logger.info(f"\t\tEntry {i}/{len(subset)}")

    if( len(lines) > 0 ):
        with open( path_partial, 'a' ) as g:
            g.write( ('\n'.join(lines) )+'\n' )

exec(subset)