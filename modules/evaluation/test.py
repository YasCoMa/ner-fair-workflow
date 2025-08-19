import os
import sys
import json
import torch
import random
import argparse
import logging

import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, LongformerTokenizerFast

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-2] )
sys.path.append( root_path )
from utils.utils_evaluation import *

class Test:
    def __init__(self):
        self.seed = 42

        self._setup_gpu()
        self._get_arguments()
        self._setup_out_folders()

        self.fready = os.path.join( self.logdir, "tasks_test.ready")
        self.ready = os.path.exists( self.fready )
        if(self.ready):
            self.logger.info("----------- Test step skipped since it was already computed -----------")
            self.logger.info("----------- Test step ended -----------")
    
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
        parser = argparse.ArgumentParser(description='Test')
        parser.add_argument('-execDir','--execution_path', help='Directory where the logs and history will be saved', required=True)
        parser.add_argument('-paramFile','--parameter_file', help='Running configuration file', required=False)
        
        args = parser.parse_args()
        execdir = args.execution_path
        self.logdir = os.path.join( execdir, "logs" )
        if( not os.path.exists(self.logdir) ):
            os.makedirs( self.logdir )
        logf = os.path.join( self.logdir, "test.log" )
        logging.basicConfig( filename=logf, encoding="utf-8", filemode="a", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' )
        self.logger = logging.getLogger('test')

        with open( args.parameter_file, 'r' ) as g:
            self.config = json.load(g)

        try:
            self.model_checkpoint = self.config["pretrained_model"]
            self.outDir = self.config["outpath"]
            self.dataDir = os.path.join(self.outDir, "preprocessing", "dataset_train_valid_test_split_v0.1") # Transformers dataset utput from preproc step
            self.target_tags = json.load( open( self.config["target_tags"], 'r') )
            
            self.logger.info("----------- Test step started -----------")
        except:
            raise Exception("Mandatory fields not found in config. file")

    def _setup_out_folders(self):
        task = 'ner'
        model_name = self.model_checkpoint.split("/")[-1]
        fout = '.'
        if self.outDir is not None:
            fout = self.outDir
        self.outDir = f"{fout}/{model_name}-finetuned-{task}"

        self.out = os.path.join( self.outDir, "test" )
        if( not os.path.exists(self.out) ):
            os.makedirs( self.out )

        self.out_report = os.path.join( self.out, "reports" )
        if( not os.path.exists(self.out_report) ):
            os.makedirs( self.out_report )

        self.out_eval = os.path.join( self.out, "evaluations" )
        if( not os.path.exists(self.out_eval) ):
            os.makedirs( self.out_eval )

    def _setup_seed(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.flag_tokenizer = None

        if( self.model_checkpoint.lower().find('longformer') != -1 ):
            self.tokenizer = LongformerTokenizerFast.from_pretrained(self.model_checkpoint, add_prefix_space=True)
            self.flag_tokenizer = 'LongFormer'
            
    def _load_input_data(self):
        self.logger.info("[Test step] Task (Loading input dataset) started -----------")
        task = 'ner'
        '''
        data_files = { 'train': '', 'valid': '', 'test': '' }
        for f in os.listdir( self.dataDir ):
            for t in data_files:
                if( (f.lower().startswith(t)) and (data_files[t] == '') ):
                    data_files[t] = f

        datasets = load_dataset('parquet', data_dir = args['data'], data_files=data_files)
        '''
        self.datasets = load_from_disk( self.dataDir )
        self.label_list = self.datasets['train'].features[f"{task}_tags"].feature.names
        idsxlabel = {i: label for i, label in enumerate( self.label_list)}
        self.labelxids = {label: i for i, label in enumerate( self.label_list)}
        
        # Preprocessing the data
        label_all_tokens = True
        print("TOKENIZING...")
        self.tokenized_datasets = self.datasets.map(tokenize_and_align_labels, batched=True, fn_kwargs={"flag_tokenizer": self.flag_tokenizer, "tokenizer": self.tokenizer, "label_all_tokens": label_all_tokens })
        self.logger.info("[Test step] Task (Loading input dataset) started -----------")

    def __annotate_samples(self, dataset, labels, criteria = 'first_label'):
        self.logger.info("[Test step] Task (Sentence annotation) started -----------")

        '''
        Annotate the sentences in the dataset with the predicted labels.

        This function takes a dataset of sentences and a corresponding list of labels, and annotates each sentence with its predicted labels.
        The labels are assigned to the words in the sentences based on the specified criteria: 'first_label' or 'majority'.
        'first_label' assigns the label of the first sub-token after tokenization to each word, while 'majority' assigns the most frequent label in the tokens into which the word has been divided.

        Args:
            dataset (list of dict): A list of dictionaries, each representing a sentence. Each dictionary contains 'tokens' and 'word_ids'.
            labels (list of list): A list of lists, each containing the predicted labels for a sentence.
            criteria (str): The criteria to use to select the label when the words pieces have different labels. 
                            Options are 'first_label' and 'majority'. Default is 'first_label'.

        Returns:
            annotated_sentences (list of list): A list of lists, each containing the annotated labels for a sentence.
        '''
        
        annotated_sentences = []
        for i in range(len(dataset)):
            # get just the tokens different from None
            sentence = dataset[i]
            word_ids = sentence['word_ids']
            sentence_labels = labels[i]
            annotated_sentence = [[] for _ in range(len(dataset[i]['tokens']))]
            for word_id, label in zip(word_ids, sentence_labels):
                if word_id is not None:
                    annotated_sentence[word_id].append(label)
            annotated_sentence_filtered = []
            if criteria == 'first_label':
                annotated_sentence_filtered = [annotated_sentence[i][0] for i in range(len(annotated_sentence)) if len(annotated_sentence[i])>0]
            elif criteria == 'majority':
                annotated_sentence_filtered = [ max(set(annotated_sentence[i]), key=annotated_sentence[i].count) for i in range(len(annotated_sentence)) if len(annotated_sentence[i])>0]

            annotated_sentences.append(annotated_sentence_filtered)
        self.logger.info("[Test step] Test (Sentence annotation) ended -----------")

        return annotated_sentences

    def _get_predictions(self):
        self.logger.info("[Test step] Task (Get predictions for test set) started -----------")
        device = self.device
        save_path = self.outDir
        tokenizer = self.tokenizer
        datasets = self.datasets
        tokenized_datasets = self.tokenized_datasets

        model_files = [file for file in os.listdir(save_path) if file.startswith("model_")]
        model_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        input_ids = torch.tensor(tokenized_datasets["test"]["input_ids"]).to(device)
        attention_mask = torch.tensor(tokenized_datasets["test"]["attention_mask"]).to(device)     
        test_data = {'input_ids': input_ids, 'attention_mask': attention_mask}

        labels = tokenized_datasets['test']['labels']

        models_predictions = []
        for i, model_file in enumerate(model_files):
            #print(i, model_file)
            print(f"EVALUATING... model {i+1}")
            model = torch.load(f"{save_path}/{model_file}", weights_only=False)
            model.to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(**test_data)

            predictions = torch.argmax(outputs.logits, dim=2).to("cpu").numpy()
            torch.cuda.empty_cache()

            # Per token
            generate_reports(predictions, labels, self.label_list, self.out_report, f"{i}_token_level", self.target_tags)

            # Per word
            annotated_samples_first = self.__annotate_samples(tokenized_datasets["test"], predictions)
            generate_reports(annotated_samples_first, datasets['test']['ner_tags'] , self.label_list, self.out_report, f"{i}_word_level", self.target_tags)
            models_predictions.append(annotated_samples_first)

        generate_csv_comparison(self.out_report, target_tags = self.target_tags)
        
        #Save predictions for the models in csv files

        # Create a dictionary for the dataset
        dataset_dict = {'tokens': tokenized_datasets["test"]['tokens'], 'file': tokenized_datasets["test"]['id'], 'true_labels': datasets['test']['ner_tags']}
        for i in range(len(model_files)):
            dataset_dict[f'Predicted_label_{i}'] = models_predictions[i]

        self.logger.info("[Test step] Task (Get predictions for test set) ended -----------")

        return dataset_dict

    def _save_most_common_predictions(self, dataset_dict):
        save_path = self.outDir
        
        self.logger.info("[Test step] Task (Save most common predicted labels for a token across the models) started -----------")
        dataset_annotated = Dataset.from_dict(dataset_dict)

        model_files = [file for file in os.listdir(save_path) if file.startswith("model_")]
        model_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        # Generate the files
        most_common_predictions = []
        for j in range(len(dataset_annotated['file'])):
            csv_dict = {'token': [], 'True label': [], 'Most common': []}
            for i in range(len(model_files)):
                csv_dict[f'Pred model {i+1}'] = []

            file_data = dataset_annotated.filter(lambda x: x['file'] == str(j))

            for data in zip(file_data['tokens'], file_data['true_labels'], *(file_data[f'Predicted_label_{i}'] for i in range(len(model_files)))):
                sentence, or_labels, *preds = data
                for token_data in zip(sentence, or_labels, *preds):
                    token, or_label, *pred_values = token_data
                    csv_dict['token'].append(token)
                    csv_dict['True label'].append( self.label_list[or_label])

                    for i, pred_value in enumerate(pred_values):
                        csv_dict[f'Pred model {i+1}'].append( self.label_list[pred_value])
                    occurence_counter = Counter([csv_dict[f'Pred model {i+1}'][-1] for i in range(len(model_files))])
                    csv_dict['Most common'].append(occurence_counter.most_common(1)[0][0])

            most_common_predictions.append(csv_dict['Most common'])
            path = os.path.join( self.out_eval, f"File_{j}_IOB.csv" )
            pd.DataFrame.from_dict(csv_dict,  orient='index').transpose().to_csv()

        # Compute the classification reports with most common predictions
        most_common_predictions = [[ self.labelxids[id] for id in lab] for lab in most_common_predictions]
        generate_reports(most_common_predictions,  dataset_annotated['true_labels'], self.label_list, self.out_report, 'most_common', self.target_tags)
        self.logger.info("[Test step] Task (Save most common predicted labels for a token across the models) ended -----------")

    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Test step ended -----------")

    def run(self):
        self._setup_seed()
        self._setup_model()
        self._load_input_data()
        dataset_dict = self._get_predictions()
        self._save_most_common_predictions(dataset_dict)
        self._mark_as_done()

if( __name__ == "__main__" ):
    i = Test( )
    if( not i.ready ):
        i.run()