import sys, os
import pandas as pd
import numpy as np
from datasets import DatasetDict

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-2] )
sys.path.append( root_path )
sys.path.append('.')
from utils_preprocessing import *
from brat.tools.anntoconll import main

#Define the tags
tag_values =["O",
            "B-total-participants",
            "I-total-participants",
            "B-intervention-participants",
            "I-intervention-participants",
            "B-control-participants",
            "I-control-participants",
            "B-age",
            "I-age",
            "B-eligibility",
            "I-eligibility",
            "B-ethinicity",
            "I-ethinicity",
            "B-condition",
            "I-condition",
            "B-location",
            "I-location",
            "B-intervention",
            "I-intervention",
            "B-control",
            "I-control",
            "B-outcome",
            "I-outcome",
            "B-outcome-Measure",
            "I-outcome-Measure",
            "B-iv-bin-abs",
            "I-iv-bin-abs",
            "B-cv-bin-abs",
            "I-cv-bin-abs",
            "B-iv-bin-percent",
            "I-iv-bin-percent",
            "B-cv-bin-percent",
            "I-cv-bin-percent",
            "B-iv-cont-mean",
            "I-iv-cont-mean",
            "B-cv-cont-mean",
            "I-cv-cont-mean",
            "B-iv-cont-median",
            "I-iv-cont-median",
            "B-cv-cont-median",
            "I-cv-cont-median",
            "B-iv-cont-sd",
            "I-iv-cont-sd",
            "B-cv-cont-sd",
            "I-cv-cont-sd",
            "B-iv-cont-q1",
            "I-iv-cont-q1",
            "B-cv-cont-q1",
            "I-cv-cont-q1",
            "B-iv-cont-q3",
            "I-iv-cont-q3",
            "B-cv-cont-q3",
            "I-cv-cont-q3"]

def _transform_format():
	#Read the text roots from the data folder
	txt_files = ['data/'+file for file in os.listdir('data/') if file.endswith(".txt")]
	#Run the brat tools on the text files to generate the conll files
	main(['-']+txt_files) #The - is a dummy argument to make the brat tools work

	#Read the root of .conll files
	conll_files = sorted([file for file in os.listdir('./data/') if file.endswith(".conll")])

	data = {'File_ID': [], 'Entity': [], 'Start': [], 'End': [], 'Words': []}
	for file in conll_files:
	    #Read the files
	    file_path = os.path.join('./data/', file)
	    with open(file_path, "r") as f:
	        lines = f.readlines()

	        #Add the info of each line of 'file' to the data dict
	        for line in lines:
	            data['File_ID'].append(file.split('.')[0])
	            line = line.split()
	            if len(line)>0:
	                data['Entity'].append(line[0])
	                data['Start'].append(line[1])
	                data['End'].append(line[2])
	                data['Words'].append(line[3])
	            else: #For blank lines
	                data['Entity'].append(np.nan)
	                data['Start'].append(np.nan)
	                data['End'].append(np.nan)
	                data['Words'].append(np.nan)

	#Transform data dict into DataFrame
	data_df = pd.DataFrame(data)
	data_df.to_csv('tmp.csv', index=None)

def _add_sentence_id_save():
	data_df = pd.read_csv('tmp.csv')

	# Create list of sentences and counter
	ls = []
	counter = 1

	for i in range(len(data_df)):
	    # If the current line is not a sentence separator, we add the current counter to the list
	    if ((pd.isna(data_df.Words[i]) == False) | (pd.isna(data_df.Entity[i]) == False)):
	        ls.append(counter)
	    # If the current line is a sentence separator, we add 0 to the list and increase the counter by 1
	    if ((pd.isna(data_df.Words[i]) == True) & (pd.isna(data_df.Entity[i]) == True)):
	        ls.append(0)
	        counter += 1 
	        
	# We add the list to the dataframe        
	data_df['Sentence_ID'] = ls
	data_df = data_df.dropna() # 11967 lines excluded
	data_df['Start'] = data_df['Start'].astype(int)
	data_df['End'] = data_df['End'].astype(int)
	data_df.to_csv('./DataProcessed/dataBIO_v3.csv', index=False)

def _split():
	data_df = pd.read_csv('./DataProcessed/dataBIO_v3.csv')

	File_ID = data_df['File_ID'].unique()
	# Create a new random generator
	rng = np.random.default_rng(20)

	# Convert File_ID to a numpy array
	File_ID_np = np.array(File_ID)

	# Generate indices for training set
	train_indices = rng.choice(len(File_ID_np), size=int(len(File_ID_np)*0.8), replace=False)

	# Get the training set
	train = File_ID_np[train_indices]

	# Get the remaining indices
	remaining_indices = np.array([i for i in File_ID_np if i not in train])

	# Assuming `train` is a numpy array containing the elements you want to compare with `File_ID_np`
	not_in_train = np.setdiff1d(File_ID_np, train)                 #For Experiment 1 split train-test

	#For Experiment 2 split train-valid-test
	# Generate indices for validation set from the remaining indices
	rng = np.random.default_rng(20)
	valid_indices = rng.choice(len(remaining_indices), size=int(len(remaining_indices)*0.5), replace=False)
	# Get the validation set
	valid = remaining_indices[valid_indices]

	# Get the test set
	test = np.array(np.setdiff1d(remaining_indices, valid))

	#Generate the dataBIO dataframes for each set
	train_df = data_df[data_df['File_ID'].isin(list(train))]
	valid_df = data_df[data_df['File_ID'].isin(list(valid))]
	test_df = data_df[data_df['File_ID'].isin(list(test))]
	not_in_train_df = data_df[data_df['File_ID'].isin(list(not_in_train))]

	return data_df, train_df, test_df, valid_df, not_in_train_df

def _gen_freq_stats(data_df, train_df, test_df, valid_df, not_in_train_df):
	#Create a copy of the dataframe
	data2_df = data_df.copy()

	#Drop the rows where the Entity is O because it is not an entity of interest
	data2_df = data2_df[data2_df['Entity'] != 'O']

	#Count values of the column 'Entity' that start with 'B-' because it determines the number of elements per entity
	Freq = pd.DataFrame(data2_df['Entity'].value_counts())
	#Drop the rows where the index starts with 'I-'
	Freq = Freq[Freq.index.str.startswith('I-') == False]

	#Add the number of fileID uniques where each label appears
	Freq['n_files'] = data2_df.groupby('Entity')['File_ID'].nunique()

	#Re-order the labels to be in the same order as in the paper mentioned
	Freq = Freq.reindex(['B-total-participants', 'B-intervention-participants','B-control-participants', 'B-age', 'B-eligibility', 'B-ethinicity', 'B-condition', 'B-location', 'B-intervention', 'B-control', 'B-outcome', 'B-outcome-Measure', 'B-iv-bin-abs', 'B-cv-bin-abs', 'B-iv-bin-percent', 'B-cv-bin-percent', 'B-iv-cont-mean', 'B-cv-cont-mean', 'B-iv-cont-median', 'B-cv-cont-median', 'B-iv-cont-sd', 'B-cv-cont-sd', 'B-iv-cont-q1', 'B-cv-cont-q1', 'B-iv-cont-q3', 'B-cv-cont-q3'])

	Freq['train_set'] = train_df[train_df['Entity'] != 'O'].value_counts('Entity')
	Freq['valid_set'] = valid_df[valid_df['Entity'] != 'O'].value_counts('Entity')
	Freq['test_set'] = test_df[test_df['Entity'] != 'O'].value_counts('Entity')
	Freq['not_in_train'] = not_in_train_df[not_in_train_df['Entity'] != 'O'].value_counts('Entity')
	Freq.to_csv("Freq_B_tags.csv")

	#Count the number of entities in each set
	counter_train = train_df['Entity'].apply(lambda x: ' '.join([word.split('-',1)[1] if word != 'O' else word for word in x.split()])).value_counts()
	counter_valid = valid_df['Entity'].apply(lambda x: ' '.join([word.split('-',1)[1] if word != 'O' else word for word in x.split()])).value_counts()
	counter_test = test_df['Entity'].apply(lambda x: ' '.join([word.split('-',1)[1] if word != 'O' else word for word in x.split()])).value_counts()
	counter_not_in_train = not_in_train_df['Entity'].apply(lambda x: ' '.join([word.split('-',1)[1] if word != 'O' else word for word in x.split()])).value_counts()
	#Count the number of entities in all the data
	counter_all = data_df['Entity'].apply(lambda x: ' '.join([word.split('-',1)[1] if word != 'O' else word for word in x.split()])).value_counts()

	#Create a dataframe with the values of the counters
	df_counter = pd.DataFrame({'all': counter_all, 'train': counter_train, 'valid': counter_valid, 'test': counter_test, 'not_in_train': counter_not_in_train})

	#Reindex the dataframe
	df_counter = df_counter.reindex(['total-participants', 'intervention-participants', 'control-participants', 'age', 'eligibility', 'ethinicity', 'condition', 'location', 'intervention', 'control', 'outcome', 'outcome-Measure', 'iv-bin-abs', 'cv-bin-abs', 'iv-bin-percent', 'cv-bin-percent', 'iv-cont-mean', 'cv-cont-mean', 'iv-cont-median', 'cv-cont-median', 'iv-cont-sd', 'cv-cont-sd', 'iv-cont-q1', 'cv-cont-q1', 'iv-cont-q3', 'cv-cont-q3'])
	df_counter.to_csv('Freq_entities.csv')
	

def _process_group_sentences(data_df, train_df, test_df, valid_df, not_in_train_df):
	# ----------------
	train_info = GenerateInfoDF(train_df)
	valid_info = GenerateInfoDF(valid_df)
	test_info = GenerateInfoDF(test_df)
	not_in_train_info = GenerateInfoDF(not_in_train_df)

	#Case of complete text
	df_train_processed = PreprocessingData(train_df)
	df_valid_processed = PreprocessingData(valid_df)
	df_test_processed = PreprocessingData(test_df)
	df_not_in_train_processed = PreprocessingData(not_in_train_df)

	#Preprocess all the data together
	df_processed = PreprocessingData(data_df)

	not_in_train_df.to_csv('./DataProcessed/test_and_validBIO_v1.csv', sep=' ', index=False) #Experiment 1
	df_not_in_train_processed.to_csv('./DataProcessed/test_and_valid_v1.csv', sep = ' ', index=False) #Experiment 1

	#Experiment 2
	train_df.to_csv('./DataProcessed/trainBIO_v02.csv', sep=' ', index=False)
	valid_df.to_csv('./DataProcessed/validBIO_v02.csv', sep=' ', index=False)
	test_df.to_csv('./DataProcessed/testBIO_v02.csv', sep=' ', index=False)

	#Experiment 2
	df_train_processed.to_csv('./DataProcessed/train_v02.csv', sep=' ', index=False)
	df_valid_processed.to_csv('./DataProcessed/valid_v02.csv', sep=' ', index=False)
	df_test_processed.to_csv('./DataProcessed/test_v02.csv', sep=' ', index=False)

	df_processed.to_csv('./DataProcessed/full_data.csv', sep=' ', index=False)

def _gen_final_dataset():
	#Load data from csv files if the notebook is restarted

	path_train = "./DataProcessed/train_v02.csv"
	path_valid = "./DataProcessed/valid_v02.csv"
	path_test  = "./DataProcessed/test_v02.csv"
	path_data  = "./DataProcessed/full_data.csv"
	path_tv    = "./DataProcessed/test_and_valid_v1.csv"

	def load_data(path):
	    #Read csv file specifying that the first row is the header
	    df = pd.read_csv(path, sep=" ", header=0)
	    return df

	train = load_data(path_train)
	valid = load_data(path_valid)
	test  = load_data(path_test)
	data  = load_data(path_data)
	not_in_train = load_data(path_tv)

	words_train, words_labels_train = SplitData(train)
	words_valid, words_labels_valid = SplitData(valid)
	words_test, words_labels_test = SplitData(test)
	words_tv, words_labels_tv = SplitData(not_in_train)

	words_data, words_labels_data = SplitData(data)

	tag2idx = {t: i for i, t in enumerate(tag_values)}
	idx2tag = {i: t for i, t in enumerate(tag_values)}

	#Map the labels to the tags
	labels_train = MapLabels(words_labels_train, tag2idx)
	labels_valid = MapLabels(words_labels_valid, tag2idx)
	labels_test  = MapLabels(words_labels_test, tag2idx)
	labels_tv    = MapLabels(words_labels_tv, tag2idx)

	labels_data  = MapLabels(words_labels_data, tag2idx)

	#Create a dataset for each set
	train_dataset = CreateDataset(words_train, labels_train, tag_values, list(train['File_ID']))
	valid_dataset = CreateDataset(words_valid, labels_valid, tag_values, list(valid['File_ID']))
	test_dataset  = CreateDataset(words_test, labels_test, tag_values, list(test['File_ID']))
	tv_dataset    = CreateDataset(words_tv, labels_tv, tag_values, list(not_in_train['File_ID']))

	data_dataset  = CreateDataset(words_data, labels_data, tag_values, list(data['File_ID']))

	dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset, "test": test_dataset})
	dataset2 = DatasetDict({"train": train_dataset, "test": tv_dataset})

	#Save the dataset
	dataset.save_to_disk("./DataProcessed/dataset_split_v0.2")
	dataset2.save_to_disk('./DataProcessed/train_test_split_v0.2')
	data_dataset.save_to_disk("./DataProcessed/dataset_v0.2")

def run():
	#_transform_format()
	#_add_sentence_id_save()
	data_df, train_df, test_df, valid_df, not_in_train_df = _split()
	_gen_freq_stats(data_df, train_df, test_df, valid_df, not_in_train_df)
	_process_group_sentences(data_df, train_df, test_df, valid_df, not_in_train_df)
	_gen_final_dataset()

run()