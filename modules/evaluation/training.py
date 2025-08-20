import os
import re
import sys
import json
import glob
import torch
import random
import pickle
import optuna
import argparse
import logging

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk
from seqeval.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, LongformerTokenizerFast, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-2] )
sys.path.append( root_path )
from utils.utils_evaluation import *

class Training:
    def __init__(self):
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.seed = 42

        self._setup_gpu()
        self._get_arguments()
        self._setup_out_folders()

        self.fready = os.path.join( self.logdir, "tasks_training.ready")
        self.ready = os.path.exists( self.fready )
        if(self.ready):
            self.logger.info("----------- Training step skipped since it was already computed -----------")
            self.logger.info("----------- Training step ended -----------")
    
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
        parser = argparse.ArgumentParser(description='Training step')
        parser.add_argument('-execDir','--execution_path', help='Directory where the logs and history will be saved', required=True)
        parser.add_argument('-paramFile','--parameter_file', help='Running configuration file', required=False)
        
        args = parser.parse_args()
        execdir = args.execution_path
        self.logdir = os.path.join( execdir, "logs" )
        if( not os.path.exists(self.logdir) ):
            os.makedirs( self.logdir )
        logf = os.path.join( self.logdir, "training.log" )
        logging.basicConfig( filename=logf, encoding="utf-8", filemode="a", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' )
        self.logger = logging.getLogger('training')

        with open( args.parameter_file, 'r' ) as g:
            self.config = json.load(g)

        try:
            self.model_checkpoint = self.config["pretrained_model"]
            self.outDir = self.config["outpath"]
            self.dataDir = os.path.join(self.outDir, "preprocessing", "dataset_train_valid_test_split_v0.1") # Transformers dataset utput from preproc step
            self.target_tags = json.load( open( self.config["target_tags"], 'r') )
            
            self.report_summary_stats_metric = 'median'
            if( 'report_summary_stats_metric' in self.config ):
                if( self.config['report_summary_stats_metric'] in ['max', 'min', 'mean', 'median', 'std'] ):
                    self.report_summary_stats_metric = self.config['report_summary_stats_metric']

            self.optimization_metric = 'f1'
            if( 'optimization_metric' in self.config ):
                if( self.config['optimization_metric'] in ['f1', 'precision', 'recall', 'accuracy'] ):
                    self.optimization_metric = self.config['optimization_metric']

            self.do_optimization = True
            self.optimization_file = None
            if( 'do_hyperparameter_search' in self.config ):
                self.do_optimization = self.config["do_hyperparameter_search"]
            if( 'hyperparameter_path' in self.config ):
                self.optimization_file = self.config["hyperparameter_path"]
                if( self.optimization_file == "" ):
                    self.optimization_file = None

            self.logger.info("----------- Training step started -----------")
        except:
            raise Exception("Mandatory fields not found in config. file")

    def _setup_out_folders(self):
        task = 'ner'
        model_name = self.model_checkpoint.split("/")[-1]
        fout = '.'
        if self.outDir is not None:
            fout = self.outDir
        self.outDir = f"{fout}/{model_name}-finetuned-{task}"

        self.out = os.path.join( self.outDir, "training" )
        if( not os.path.exists(self.out) ):
            os.makedirs( self.out )

        self.out_report = os.path.join( self.out, "reports" )
        if( not os.path.exists(self.out_report) ):
            os.makedirs( self.out_report )

        self.out_agg_report = os.path.join( self.out, "summary_reports" )
        if( not os.path.exists(self.out_agg_report) ):
            os.makedirs( self.out_agg_report )

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
        self.logger.info("[Training step] Task (Loading input dataset) started -----------")
        task = 'ner'

        '''
        data_files = { 'train': '', 'valid': '', 'test': '' }
        for f in os.listdir( self.dataDir ):
            for t in data_files:
                if( (f.lower().startswith(t)) and (data_files[t] == '') ):
                    data_files[t] = f

        datasets = load_dataset('parquet', data_dir = args['data'], data_files=data_files)
        '''
        datasets = load_from_disk( self.dataDir )
        self.label_list = datasets['train'].features[f"{task}_tags"].feature.names
        self.idsxlabel = {i: label for i, label in enumerate( self.label_list)}
        self.labelxids = {label: i for i, label in enumerate( self.label_list)}
        
        # Preprocessing the data
        label_all_tokens = True
        print("TOKENIZING...")
        self.tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, fn_kwargs={"flag_tokenizer": self.flag_tokenizer, "tokenizer": self.tokenizer, "label_all_tokens": label_all_tokens } )
        self.logger.info("[Training step] Task (Loading input dataset) ended -----------")

    def _manage_optimization(self):
        self.logger.info("[Training step] Task (Manage hyperparameter optimization) started -----------")
        device = self.device
        BATCH_SIZE = self.batch_size
        LEARNING_RATE = self.learning_rate
        WEIGHT_DECAY = self.weight_decay
        save_path = self.out
        tokenizer = self.tokenizer
        tokenized_datasets = self.tokenized_datasets

        HYPERPARAMETERS_SEARCH = self.do_optimization
        pkl_file = glob.glob(os.path.join(save_path, "*.pkl"))
        if self.optimization_file != None:
            #Case when hyperparameters are provided
            with open( self.optimization_file, 'rb') as file:
                self.hyperparameters_loaded = pickle.load(file)
        elif pkl_file:
            #Case when hyperparameter optimization has already been done by the script
            with open(pkl_file[0], 'rb') as file:
                self.hyperparameters_loaded = pickle.load(file)
        elif HYPERPARAMETERS_SEARCH:
            #Hyperparameter fine-tuning
            print("FINE-TUNING HYPERPARAMETERS...")
            def objective(trial: optuna.Trial):
                model = AutoModelForTokenClassification.from_pretrained( self.model_checkpoint, num_labels=len(self.label_list), label2id=self.labelxids, id2label=self.idsxlabel)
                print(f"Trial {trial.number}")   
                model.to(device)

                if( self.model_checkpoint.find('longformer') != -1 ):
                    hyperparameters_space = {'per_device_train_batch_size': trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
                                    'per_device_eval_batch_size': trial.suggest_categorical("per_device_eval_batch_size", [8, 16])}
                else:
                    hyperparameters_space = {'per_device_train_batch_size': trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
                                'per_device_eval_batch_size': trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32])}


                hpath = os.path.join(self.out, 'hyperparameter_search')
                args = TrainingArguments(
                hpath,
                learning_rate=trial.suggest_float("learning_rate", low=2e-5, high=5e-5, log=True),
                weight_decay=trial.suggest_float("weight_decay", 4e-5, 0.01, log=True),
                num_train_epochs=10,
                eval_strategy = "epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                greater_is_better = False,
                eval_accumulation_steps=1,
                **hyperparameters_space
                )

                data_collator = DataCollatorForTokenClassification(tokenizer)

                trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['valid'],
                data_collator=data_collator,
                tokenizer=tokenizer,
                )
            
                trainer.train()
                if device.type == 'cuda':
                    #print(torch.cuda.get_device_name(0))
                    print('Memory Usage:')
                    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

                #We want to maximize the f1-score in validation set
                predictions, labels, metrics = trainer.predict(tokenized_datasets['valid'])  
                print(f"Validation set metrics: \n {metrics}")
                metric_value = __compute_objective(predictions, labels, self.label_list, self.optimization_metric)
                print(f"{self.optimization_metric}: {metric_value}")
                return metric_value

            # We want to maximize the f1-score in validation set
            study = optuna.create_study(study_name="hyper-parameter-search", direction="maximize")
            study.optimize(func=objective, n_trials=15)
            print(f"Best F1-score: {study.best_value}")
            print(f"Best parameters: {study.best_params}")
            #print(f"Best trial: {study.best_trial}")

            file = open(f"{save_path}/best_params.pkl", "wb")
            pickle.dump(study.best_params, file)
            file.close()
            self.hyperparameters_loaded = study.best_params
        
        else:
            #Case when hyperparameter optimization is not wanted and has been defined manually or in Experiment 1
            self.hyperparameters_loaded = {'learning_rate': LEARNING_RATE,
                                    'weight_decay': WEIGHT_DECAY,
                                    'per_device_train_batch_size': BATCH_SIZE,
                                    'per_device_eval_batch_size': BATCH_SIZE
                                    }
        self.logger.info("[Training step] Task (Manage hyperparameter optimization) ended -----------")

    def __generate_learning_curves(self, path_data):
        self.logger.info("[Training step] Task (Generating learning curve plots) started -----------")
        '''
        Generate learning curves plots and F1-scores evolution during training on validation set.

        This function reads the training log files, extracts the loss and F1-score values, and generates plots of these metrics over epochs.
        It creates a plot for each training log file, showing the training and validation loss over epochs.
        It also creates a single plot showing the F1-score on the validation set over epochs for all training runs.
        The plots are saved as PNG files in the specified directory.

        Args:
            path_data (str): Path to the directory where the training log files are located and where the output PNG files will be saved.

        Returns:
            None. The function saves the plots as PNG files in the path_data directory.
        '''
        #Read the data
        files = [file for file in os.listdir(path_data) if file.startswith("log_history")]
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            
        eval_f1 = pd.DataFrame()
        
        for i, file in enumerate(files):
            plt.figure()
            data = pd.read_csv(f"{path_data}/{file}", delimiter=',')

            #Get the entries of the loss column that are not NaN
            loss = data['loss'].dropna().reset_index(drop=True)
            val_loss = data['eval_loss'].dropna().reset_index(drop=True)

            #Get the entries of the eval_f1 column that are not NaN
            eval_f1[f"Training {i}"] = data['eval_f1'].dropna().reset_index(drop=True)
            
            #Plot the loss and val_loss
            plt.plot(loss, label=f'Train loss')
            plt.plot(val_loss, label=f'Val loss')
            plt.legend()
            plt.title(f"Loss on training and validation set over epochs training {i}")
            #Save the plot
            plt.savefig(f"{path_data}/loss_{i}.png")
            plt.close()

        #Get the epochs where the weights are restored and the f1-score
        epoch_save = list(eval_f1.idxmax())
        max_f1     = list(eval_f1.max())    
        #Plot the eval_f1
        eval_f1.plot(legend=False)
        plt.scatter(np.array(epoch_save), np.array(max_f1), s=5, c='red')
           
        #plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title("F1-score on validation set over epochs")
        plt.savefig(f"{path_data}/eval_f1.png")
        plt.close()
        self.logger.info("[Training step] Task (Generating learning curve plots) ended -----------")

    def _train(self):
        self.logger.info("[Training step] Task (Training model) started -----------")
        device = self.device
        BATCH_SIZE = self.batch_size
        LEARNING_RATE = self.learning_rate
        WEIGHT_DECAY = self.weight_decay
        save_path = self.outDir
        tokenizer = self.tokenizer
        label_list = self.label_list
        tokenized_datasets = self.tokenized_datasets
        hyperparameters_loaded = self.hyperparameters_loaded

        def compute_metrics(p):
            '''
            Compute accuracy, F1 score, and classification report for the given predictions and labels.

            This function takes a tuple of predictions and labels, converts the predictions to labels, filters out special tokens,
            and then computes the accuracy, F1 score, and classification report. The results are returned in a dictionary.
    
            Args:
                p (tuple): A tuple containing two elements:
                    - predictions (numpy.ndarray): A 2D array of predicted probabilities for each label.
                    - labels (list of list): A list of lists, each containing the true labels for a sentence.

            Returns:
                results (dict): A dictionary containing the following metrics:
                    - 'accuracy': The accuracy of the predictions.
                    - 'f1': The F1 score of the predictions.
                    - 'classification_report': The classification report, computed in 'strict' mode with the IOB2 scheme.
            '''
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
           
            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = {
                  'accuracy': accuracy_score(true_labels, true_predictions),
                  'f1': f1_score(true_labels, true_predictions, zero_division=0),
                  #'classification_report': classification_report(true_labels, true_predictions, mode='strict', scheme= IOB2, output_dict=True, zero_division=0)    
            }
            return results

        #In case of providing a .pkl file without any of the four hyperparameters considered we set them to the ones of Experiment 1
        if 'per_device_train_batch_size' not in list(hyperparameters_loaded.keys()):
            hyperparameters_loaded['per_device_train_batch_size'] = BATCH_SIZE
        if 'per_device_eval_batch_size' not in list(hyperparameters_loaded.keys()):
            hyperparameters_loaded['per_device_eval_batch_size']  = BATCH_SIZE
        if 'learning_rate' not in list(hyperparameters_loaded.keys()):
            hyperparameters_loaded['learning_rate'] = LEARNING_RATE
        if 'weight_decay' not in list(hyperparameters_loaded.keys()):
            hyperparameters_loaded['weight_decay'] = WEIGHT_DECAY
        
        self.logger.info(f"\tTraining with hyperparameters: \n {hyperparameters_loaded}")
        #TRAINING
        report_identifier = 'training-model'

        for i in range(5):
            self.logger.info("\t\tTRAINING... round {i}")
            #Define the model
            model = AutoModelForTokenClassification.from_pretrained( self.model_checkpoint, num_labels=len( self.label_list), label2id=self.labelxids, id2label=self.idsxlabel)
    
            model.to(device)

            model_name = self.model_checkpoint.split("/")[-1]

            args = TrainingArguments(
                f"{save_path}/trained_model_{i}",
                num_train_epochs=40,
                eval_strategy = "epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                metric_for_best_model="f1",
                load_best_model_at_end=True,
                save_total_limit=1,
                seed=random.randint(0,200),
                eval_accumulation_steps=1,
                **hyperparameters_loaded
            )

            data_collator = DataCollatorForTokenClassification(tokenizer)

            trainer = Trainer(
                model,
                args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['valid'],
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            #Train the model
            outputs_train = trainer.train()
            if device.type == 'cuda':
                #print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

            torch.save(model, f"{save_path}/model_{i}.pt") #Save the model
            pd.DataFrame(trainer.state.log_history).to_csv(f"{save_path}/log_history_{i}.csv") #Save the logs of the training

            #Evaluate the model
            self.logger.info("\t\tEVALUATING... round{i}")
            outputs, labels, _ = trainer.predict(tokenized_datasets["test"])
            predictions = np.argmax(outputs, axis=2)

            generate_reports_table( outputs, predictions, labels, label_list, self.out_report, report_identifier, index=f'{i}', level='token' )
            #generate_reports(predictions, labels, self.label_list, self.out_report, f"{i}_token_level", self.target_tags)
        
        #generate_csv_comparison( self.out_report, type_level=['token_level'])
        entry_point = self.out_report
        out_path = self.out_agg_report
        aggregate_reports(entry_point, report_identifier, out_path, agg_stats_metric = self.report_summary_stats_metric, levels = ['word', 'token'])

        self.logger.info("[Training step] Task (Training model) ended -----------")

        self.__generate_learning_curves(save_path)

    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Training step ended -----------")

    def run(self):
        self._setup_seed()
        self._setup_model()
        self._load_input_data()
        self._manage_optimization()
        self._train()
        self._mark_as_done()

if( __name__ == "__main__" ):
    i = Training( )
    if( not i.ready ):
        i.run()