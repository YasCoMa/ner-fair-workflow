import os
import sys
import json
import torch
import random
import argparse
import logging

import pandas as pd
from uuid import uuid4
from transformers import pipeline
from transformers import AutoTokenizer, LongformerTokenizerFast, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from spacy.lang.en import English

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-2] )
sys.path.append( root_path )
from utils.commons import *
from utils.utils_evaluation import *

#sys.path.append("/aloy/data/utilities/labresource_update_singularity/pipeline-update-uniprot-ppidb/")
#from updatedbs.util.commons import *

class Prediction:
    def __init__(self):
        self.seed = 42

        self._setup_gpu()
        self._get_arguments()
        self._setup_out_folders()

        self.fready = os.path.join( self.logdir, f"{self.expid}-tasks_prediction.ready")
        self.ready = os.path.exists( self.fready )
        if(self.ready):
            self.logger.info("----------- Prediction step skipped since it was already computed -----------")
            self.logger.info("----------- Prediction step ended -----------")

    def _setup_gpu(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(f'Using device: {device}')

        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    def _get_arguments(self):
        parser = argparse.ArgumentParser(description='Prediction')
        parser.add_argument('-execDir','--execution_path', help='Directory where the logs and history will be saved', required=True)
        parser.add_argument('-paramFile','--parameter_file', help='Running configuration file', required=False)
        
        args = parser.parse_args()
        
        with open( args.parameter_file, 'r' ) as g:
            self.config = json.load(g)

        execdir = args.execution_path
        self.logdir = os.path.join( execdir, "logs" )
        if( not os.path.exists(self.logdir) ):
            os.makedirs( self.logdir )

        self.expid = self.config["identifier"]
        logf = os.path.join( self.logdir, f"{self.expid}-prediction.log" )
        logging.basicConfig( filename=logf, encoding="utf-8", filemode="a", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
        self.logger = logging.getLogger('prediction')

        try:
            self.expid = self.config["identifier"]
            self.config_path = None
            self.flag_parallel = False
            if( 'config_hpc' in self.config ):
                self.config_path = self.config['config_hpc']
                self.flag_parallel = True
            
            if( 'seed' in self.config ):
                if( isinstance( self.config['seed'] , int ) ):
                    self.seed = self.config['seed']

            self.model_checkpoint = self.config["pretrained_model"]
            self.outDir = self.config["outpath"]
            self.inpath = self.config["input_prediction"]
            self.infile = None
            self.indir = None
            if( os.path.isfile(self.inpath) ):
                self.infile = self.inpath
            elif( os.path.isdir(self.inpath) ):
                self.indir = self.inpath
            self.logger.info("----------- Prediction step started -----------")
        except:
            raise Exception("Mandatory fields not found in config. file")

    def _setup_out_folders(self):
        task = 'ner'
        model_name = self.model_checkpoint.split("/")[-1]
        fout = '.'
        if self.outDir is not None:
            fout = self.outDir
        self.outDir = os.path.join(fout, f"{self.expid}-{model_name}-finetuned-{task}" )

        self.out = os.path.join( self.outDir, "prediction" )
        if( not os.path.exists(self.out) ):
            os.makedirs( self.out )

    def _setup_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
            
    def __read_txt_file(self, file_path):
        """
        Reads the content of a text file.

        This function opens a file in read mode, reads the content, and returns it. 
        If the file does not exist, it prints an error message and returns None.

        Args:
            file_path (str): The path to the file to be read.

        Returns:
            str: The content of the file as a string if the file exists. 
            None: If the file does not exist.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return None

    def _load_input_data(self):
        self.logger.info("[Prediction step] Task (Loading input dataset) started -----------")
        nlp = English()
        nlp.add_pipe('sentencizer')

        indata = {}
        if(self.indir is not None):
            for f in os.listdir(self.indir):
                if( f.endswith('.txt') ):
                    path = os.path.join(self.indir, f)
                    file_name = path.split("/")[-1].split(".")[0]
                    raw_text = self.__read_txt_file(path)
                    #doc = nlp(raw_text)
                    #sentences = [sent.text.strip() for sent in doc.sents]
                    
                    sentences = [raw_text]
                    indata[file_name] = {}
                    for s in sentences:
                        sid = str(uuid4())
                        indata[file_name][sid] = s
        elif(self.infile is not None):
            path = self.infile
            file_name = path.split("/")[-1].split(".")[0]
            raw_text = self.__read_txt_file(path)
            #doc = nlp(raw_text)
            #sentences = [sent.text.strip() for sent in doc.sents]
            
            sentences = [raw_text]
            indata[file_name] = {}
            for s in sentences:
                sid = str(uuid4())
                indata[file_name][sid] = s

        self.logger.info("[Prediction step] Task (Loading input dataset) ended -----------")

        self.input = indata

    def _load_input_data_parallel(self):
        self.logger.info("[Prediction step] Task (Loading input dataset) started -----------")
        nlp = English()
        nlp.add_pipe('sentencizer')

        indata = {}
        if(self.indir is not None):
            for f in os.listdir(self.indir):
                if( f.endswith('.txt') ):
                    path = os.path.join(self.indir, f)
                    file_name = path.split("/")[-1].split(".")[0]
                    
                    sentences = [path]
                    indata[file_name] = {}
                    for s in sentences:
                        sid = str(uuid4())
                        indata[file_name][sid] = s
        elif(self.infile is not None):
            path = self.infile
            file_name = path.split("/")[-1].split(".")[0]
            
            sentences = [path]
            indata[file_name] = {}
            for s in sentences:
                sid = str(uuid4())
                indata[file_name][sid] = s

        self.logger.info("[Prediction step] Task (Loading input dataset) ended -----------")

        self.input = indata

    def _load_models(self):
        save_path = self.outDir
        model_files = []
        aux_directories = [os.path.join(save_path, d) for d in os.listdir(save_path) if d.startswith('trained_')]
        aux_directories.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        for directory in aux_directories:
            model_files.append([os.path.join(directory, d) for d in os.listdir(directory)][0])

        self.models = model_files

    def _get_predictions_sequential(self):
        self.logger.info("[Prediction step] Task (Get predictions for new data) started -----------")

        keys_order = ['score', 'start', 'end','entity_group', 'word']
        for i, model_file in enumerate( self.models ):
            self.logger.info(f"\tPredicting using model {i+1}")        
            classifier = pipeline("ner", model=model_file, aggregation_strategy = 'average')

            path = os.path.join( self.out, f'results_model_{i}.txt' )
            f = open( path, 'w')
            #f.write('\t'.join([key for key in ['input_file', 'sentence_id', 'sentence']+keys_order])+'\n')
            f.write('\t'.join([key for key in ['input_file']+keys_order])+'\n')
            f.close()

            i = 0
            for inf in self.input:
                sentences = self.input[inf]
                for sid in sentences:
                    st = sentences[sid]
                    try:
                        predictions = classifier(st)

                        with open( path, 'a') as f:
                            for item in predictions:
                                item['word'] = item['word'].replace('"','').replace("'",'')
                                #f.write('\t'.join( [inf, sid, st]+[str(item.get(key, '')) for key in keys_order])+'\n')
                                f.write('\t'.join( [inf]+[str(item.get(key, '')) for key in keys_order])+'\n')
                    except:
                        pass
                i += 1
                if( i%100 == 0 ):
                    self.logger.info(f"\t\tEntry {i}/{len(self.input)}")

        self.logger.info("[Prediction step] Task (Get predictions for new data) ended -----------")

    def _get_predictions_parallel(self):
        self.logger.info("[Prediction step] Task (Get predictions for new data) started -----------")

        keys_order = ['score', 'start', 'end','entity_group', 'word']
        for i, model_file in enumerate( self.models ):
            self.logger.info(f"\tPredicting using model {i+1}")        
            classifier = pipeline("ner", model=model_file, aggregation_strategy = 'average')

            path = os.path.join( self.out, f'results_model_{i}.tsv' )
            if( not os.path.isfile(path) ):
                f = open( path, 'w')
                #f.write('\t'.join([key for key in ['input_file', 'sentence_id', 'sentence']+keys_order])+'\n')
                f.write('\t'.join([key for key in ['input_file']+keys_order])+'\n')
                f.close()

                elements = []
                for inf in self.input:
                    text_files = self.input[inf]
                    for sid in text_files:
                        st = text_files[sid]
                        elements.append( [path, inf, sid, st, model_file] )

                self.logger.info("\t\tLaunching job to cluster")
                job_name = f"predictions_parallel_model_{i}"
                job_path = os.path.join( self.out, job_name )
                chunk_size = 1000
                script_path = os.path.join(os.path.dirname( os.path.abspath(__file__)), '_aux_prediction.py')
                command = "python3 "+script_path
                config = self.config_path
                prepare_job_array( job_name, job_path, command, filetasksFolder=None, taskList=elements, chunk_size=chunk_size, ignore_check = True, wait=True, destroy=True, execpy='python3', hpc_env = 'slurm', config_path=config )

                test_path_partial = os.path.join( job_path, f'part-task-1.tsv' )
                if( os.path.exists(test_path_partial) ):
                    path_partial = os.path.join( job_path, f'part-task-*.tsv' )
                    cmdStr = 'for i in '+path_partial+'; do cat $i; done | sort -u >> '+path
                    execAndCheck(cmdStr)

                    cmdStr = 'for i in '+path_partial+'; do rm $i; done '
                    execAndCheck(cmdStr)

        self.logger.info("[Prediction step] Task (Get predictions for new data) ended -----------")

    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Prediction step ended -----------")

    def run(self):
        self._setup_seed()
        self._load_models()
        
        if( not self.flag_parallel ):
            self._load_input_data()
            self._get_predictions_sequential()
        else:
            self._load_input_data_parallel()
            self._get_predictions_parallel()

        self._mark_as_done()

if( __name__ == "__main__" ):
    i = Prediction( )
    if( not i.ready ):
        i.run()