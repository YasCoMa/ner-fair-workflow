import os
import sys
import pickle
import logging
from transformers import pipeline

task_id = sys.argv[1] 
task_file = sys.argv[2]
subset = pickle.load(open(task_file, 'rb'))[task_id]

def exec(subset):
    model_file = subset[0][-1]
    classifier = pipeline("ner", model=model_file, aggregation_strategy = 'average')
    
    logging.basicConfig( format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO )
    logger = logging.getLogger( f'Prediction with model {model_file}')
    
    keys_order = ['score', 'start', 'end','entity_group', 'word']
    i=0
    for el in subset:
        path, fname, sentence_id, text_file, model_file = el
        try:
            text = open(text_file).read()
            predictions = classifier(text)
            path_partial = path.split('.')[0]+f'-part-task-{task_id}.tsv'
            with open( path_partial, 'a') as f:
                for item in predictions:
                    #f.write('\t'.join( [inf, sid, st]+[str(item.get(key, '')) for key in keys_order])+'\n')
                    f.write('\t'.join( [fname]+[str(item.get(key, '')) for key in keys_order])+'\n')
        except:
            pass
        i += 1
        if( i%100 == 0 ):
            logger.info(f"\t\tEntry {i}/{len(subset)}")

exec(subset)