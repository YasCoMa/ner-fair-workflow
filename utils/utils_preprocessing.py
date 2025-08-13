import os
from datasets import Dataset, Features, Value, Sequence, ClassLabel
import pandas as pd

'''
Script with functions to preprocess the data and analyze the mismatch between the data and the .ann files
Author: Carlos Cuevas Villarmin
Last update: 31/05/2024
'''

def ReadFiles(folder_path):
    '''
    Function that read the content of data files from a folder
    Args:
        folder_path: The path to the folder containing the data files
    Returns:
        A dictionary with the content of the id of the files, the conll, txt and ann files content
    '''
    txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]
    ann_files = [file for file in os.listdir(folder_path) if file.endswith(".ann")]
    conll_files = [file for file in os.listdir(folder_path) if file.endswith(".conll")]

    files = {'id': [], 'conll': [], 'txt': [], 'ann': [], 'id_2': [], 'id_3': []}

    for file in ann_files:
        files['id'].append(file.split('.')[0])
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            files['ann'].append(f.read())
    
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            files['txt'].append(f.read())

    for file in conll_files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            files['conll'].append(f.read())

    
    return files

def CountNumberFiles(files, B_label, ann_label, data_df, Freq, feedback=True):
    '''
    Function that analyzes the occurrence of the labels in the data and the frequency of the labels in the .ann files
    Args:
        files: dictionary with the ids, the text, the .ann and the .conll files
        B_label: label in the DataFrame (BIO format)
        ann_label: label in the .ann files
        Freq: DataFrame with the frequency of the labels in the data
    Returns:
        dictionary with the frequency of the label in the DataFrame, 
                        the number of files where the label is in the .ann files,
                        the number of files where the label is in the DataFrame,
                        the ids of the files where the label is in the .ann files,
                        the ids of the files where the label is in the DataFrame
    '''
    #Get the files['id'] where ann_label is in files['ann'] 
    id_mask = list(map(lambda file: any((ann_label == str(line.split('\t')[1].split()[0])) for line in file.split('\n') if len(line) > 1), files['ann']))
                                                                                        
    #Get the ids where id_mask is True                                                  
    from itertools import compress                                                      
    ids_ann_files = list(compress(files['id'], id_mask))
    
    #Get the ids where B_label is in data['label']
    ids_bio_format = list(data_df['File_ID'][data_df['Entity'] == B_label].unique())

    if feedback:
        if len(ids_ann_files) != len(ids_bio_format):
            print(f"The number of files is DIFFERENT.")
            print(f"Number of files where {ann_label} is in the .ann file: {len(ids_ann_files)}")
            print(f"Number of files where {B_label} is in the data: {len(ids_bio_format)}")
        else:
            print(f"The number of files is EQUAL.")
        print(f"Frequency of {B_label} in the data: {Freq.loc[B_label]['count']}")

    return {'count': Freq.loc[B_label]['count'], 'n_ann_files': len(ids_ann_files), 'n_dataframe': len(ids_bio_format), 'ids_ann_files': ids_ann_files, 'ids_bio_format': ids_bio_format}


def MismatchAnalysis(files, B_label, I_label, ann_label, data_df, freq_values, folder_path, mode = None, feedback = False):
    '''
    Function that analyzes the mismatch of the entities in the .ann files and the DataFrame
    Args:
        files: dictionary with the ids, the text and the .ann files
        B_label: label in the DataFrame (BIO format)
        I_label: label in the DataFrame (BIO format)
        ann_label: label in the .ann files
        freq_values: dictionary with the frequency of the label in the DataFrame
        folder_path: path where the .ann files are located
        mode: mode to use to analyze the mismatch. 
                'inside': analyze the mismatch inside the files that are common to both lists
                'None': analyze the mismatch of the files that just belong to one of the lists, i.e., bad transcriptions from one format to other
        feedback: boolean to print the information of the files
    Returns:
        dictionary with the ids where there exists a mismatch and the list where they belong (ids_ann_files or ids_bio_format)
    '''
    ids_ann_files = freq_values['ids_ann_files']
    ids_bio_format = freq_values['ids_bio_format']
    #Get the ids that just belong to one of the lists
    ids_mismatched = list(set(ids_ann_files) ^ set(ids_bio_format)) #^ is the symmetric difference
    #print(f"FileId that just belong to one of the lists: {ids_mismatched}")
    print(f"Number of fileIds where {ann_label} is not identified: {len(ids_mismatched)}")

    #Now to which list each id belongs
    #Create a dictionary with the ids and the list where they belong
    ids_dict = dict.fromkeys(ids_mismatched)
    for id in ids_dict:
        if id in ids_ann_files:
            ids_dict[id] = 'ids_ann_files'
        else:
            ids_dict[id] = 'ids_bio_format'

    #print(f"Dictionary with the ids and the list where they belong: {ids_dict}")

    #For each key in the dictionary read and print the corresponding file (in the folder ./data)
    from itertools import compress
    counter_mismatch = 0
    for id in ids_dict:
        if feedback: print(f"File: {id}")
        if ids_dict[id] == 'ids_ann_files':
            file_path = os.path.join(folder_path, id + '.ann')
            with open(file_path, "r") as f:
                lines = f.readlines()
                ids_lines_interest = list(map(lambda x: ann_label in x.split(), lines))
                if feedback: 
                    print(f"Lines of interest: {list(compress(lines, ids_lines_interest))}")
                counter_mismatch += len(list(compress(lines, ids_lines_interest)))
        if ids_dict[id] == 'ids_bio_format':
            if feedback: print(data_df.loc[(data_df['File_ID'] == id) & ((data_df['Entity'] == B_label) | (data_df['Entity'] == I_label))])
            counter_mismatch += len(data_df.loc[(data_df['File_ID'] == id) & ((data_df['Entity'] == B_label))]) #Counter based only in B- entities
        
    if feedback:
        #Choose one id in files['id'] that does not belong to ids_mismatched and print the lines of interest of this .ann file to compare 
        id_not_mismatched = list(set(files['id']) - set(ids_mismatched))[0]
        file_path = os.path.join(folder_path, id_not_mismatched + '.ann')
        with open(file_path, "r") as f:
            lines = f.readlines()
            ids_lines_interest = list(map(lambda x: ann_label in x, lines))
            print(f"Lines of interest of {id_not_mismatched} (no mismatch): {list(compress(lines, ids_lines_interest))}")

    print(f"Mismatched amount of entities: {counter_mismatch}")

    ######## Inside the files that are common to both lists ########
    # This requires more computational time to analyze the mismatch so it is optional and defined separately to not be executed in unnecessary cases

    if mode == 'inside':
        #Get the ids that appear in both lists
        ids_common = list(set(ids_ann_files).intersection(set(ids_bio_format)))

        common_files_mismatch_count = {'ids_common': ids_common, 'count_data': [], 'count_ann': [], 'mismatch': []}
    
        for id_file in ids_common:
            
            #Count the times B_label appears in the data in each file of ids_common
            common_files_mismatch_count['count_data'].append(data_df.loc[(data_df['File_ID'] == id_file) & (data_df['Entity'] == B_label)].shape[0])
            #Count the times ann_label appears in the .ann files in each file of ids_common 
            #Splitted in to times to avoid identifying the label in the middle of a words tagged
            aux = files['ann'][files['id'].index(id_file)].splitlines()
            counter_ann_label = 0
            for line in aux:
                splited_line = line.split()
                
                if ann_label == splited_line[1]:
                    counter_ann_label += 1
                    
            common_files_mismatch_count['count_ann'].append(counter_ann_label)
            #Count the mismatch between the times B_label appears in the data and the times ann_label appears in the .ann files in each file of ids_common
            common_files_mismatch_count['mismatch'].append(abs(common_files_mismatch_count['count_data'][-1] - common_files_mismatch_count['count_ann'][-1]))

        #Sum the values of mismatch
        total_mismatch = sum(common_files_mismatch_count['mismatch'])
        if total_mismatch != 0:
            #Get the index where the mismatch is different from 0
            keys_mismatch = [i for i, x in enumerate(common_files_mismatch_count['mismatch']) if x != 0]
            total_files_mismatch = len(keys_mismatch)
            print(f"Number of files where the occurrence in .ann files and the ocurrence in the data is not the same: {total_files_mismatch}")
            print(f"Mismatch between the number of times {B_label} appears in the data and the number of times {ann_label} appears in the .ann files: {total_mismatch}")
        else:
            print(f"The number of times {B_label} appears in the data and the number of times {ann_label} appears in the .ann files is the same.")
        
        print(f"Total entities badly transcribed: {counter_mismatch + total_mismatch}")

        #Add the keys from common_files_mismatch_count to ids_dict when mismatch is different from 0
        if total_mismatch != 0:
            for key in keys_mismatch:
                ids_dict[common_files_mismatch_count['ids_common'][key]] = 'mismatch'
    return ids_dict




def plotMismatchfiles(mismatch_dict, data_df, folder_path, control=False, feedback='all'):
    '''
    Function that prints the files where the mismatch is present
    Args:
        mismatch_dict: dictionary with the ids where there exists a mismatch and the list where they belong (ids_ann_files or ids_bio_format)
        folder_path: path where the .ann files are located
        control: boolean to stop the print
        feedback: type of mismatch to print
    '''
    for key in mismatch_dict.keys():
        #Read the file
        if len(mismatch_dict[key]) > 0:
            print(f"\n{key}")
            print('------------------------------------------------------------------------------------------')
            for file, id in zip(mismatch_dict[key].keys(),mismatch_dict[key].values()):
                if (feedback == 'all' or feedback == 'absolutely_mismatch') and id != 'mismatch':
                    print(f"\nFile: {file} absolutely poorly transcribed in terms of the entity")
                
                elif (feedback == 'all' or feedback == 'partially_mismatch') and id == 'mismatch':
                    print(f"\nFile: {file} partially poorly transcribed in terms of the entity")
                    
                file_path = os.path.join(folder_path, file + '.ann')
                with open(file_path, "r") as f:
                    print(f.read())

                if id == 'mismatch': print(data_df.loc[(data_df['File_ID'] == file) & ((data_df['Entity'] == 'B-'+ key) | (data_df['Entity'] == 'I-'+ key))])
                    
                #Input to stop the print by the user and option to break the loop
                if control:
                    input_ = input('Do you want to stop the print? (y/n): ')
                    if input_ == 'y':
                        break
        if control and input_ == 'y':
            break

def GenerateInfoDF(df):
    '''
    Function that generates a DataFrame with different metrics to analyse the similarities between samples in the sets.
    Args:
        df: DataFrame with columns File_ID, Entity, Start, End, Words, Sentence_ID
    Returns:
        df_info: DataFrame with the metrics computed
                n_tokens: number of tokens per file
                n_sentences: number of sentences per file
                n_entities: number of entities per file
                n_unique_entities: number of unique entities per file
                ratio_entities_sentence: ratio of entities per sentence
                ratio_entities_token: ratio of entities per token

    '''
    df_info =pd.DataFrame(df.groupby('File_ID')['Sentence_ID'].nunique())
    #Rename the column
    df_info = df_info.rename(columns={'Sentence_ID':'n_sentences'})
    df_info['n_tokens'] = df.groupby('File_ID')['Words'].count()
    #Add the number of entities of interest in BIO format (keep B- and I-, drop O)
    df_info['n_entities'] = df.groupby('File_ID')['Entity'].count()-df.groupby('File_ID')['Entity'].apply(lambda x: x.str.startswith('O').sum())
    #Add the number of unique entities of interest in BIO format (keep B-, drop O and I- are the same entity as B-)
    df_info['n_unique_entities'] = df.groupby('File_ID')['Entity'].apply(lambda x: x[x.str.startswith('B-')].nunique())
    #Add the ratio of entities (B- and I-) per sentence
    df_info['ratio_entities_sentence'] = df_info['n_entities'] / df_info['n_sentences']
    #Add the ratio of entities (B- and I-) per token
    df_info['ratio_entities_token'] = df_info['n_entities'] / df_info['n_tokens']

    return df_info

def PreprocessingData(df, entry_param = 'complete'):
    '''
    Function that preprocesses the data to have a dataframe with the sentences and the labels of the entities in BIO format.
    Args:
        df: dataframe with the data
            The input dataframe must have the columns 'words', 'start', 'end', 'label' 'fileId' (if entry_param='complete') and 'sentenceID' (if entry_param = 'sentence')
        entry_param: parameter that determines if the dataframe has the complete text as a unique sample or the sentences of the text separately
    Returns:
        df_processed: dataframe with the sentences and the labels of the entities in BIO format
    '''
    if entry_param == 'complete':
        #Add a column with the sentence that each word belongs to
        df['Sentence'] = df.groupby(['File_ID'])['Words'].transform(lambda x: ' '.join(x))
        #Add a column with all the labels of the words that belong to the same sentence
        df['Entity_sentence'] = df.groupby(['File_ID'])['Entity'].transform(lambda x: ' '.join(x))

        df_processed = df[['File_ID', 'Sentence', 'Entity_sentence']].drop_duplicates().reset_index(drop=True)
    
    elif entry_param == 'sentence':
        #Add a column with the sentence that each word belongs to
        df['Sentence'] = df.groupby(['Sentence_ID'])['Words'].transform(lambda x: ' '.join(x))
        #Add a column with all the labels of the words that belong to the same sentence
        df['Entity_sentence'] = df.groupby(['Sentence_ID'])['Entity'].transform(lambda x: ' '.join(x))

        df_processed = df[['Sentence_ID', 'Sentence', 'Entity_sentence']].drop_duplicates().reset_index(drop=True)

    return df_processed

def SplitData(df):
    '''
    Function that splits the data into words and labels
    Args:   
        df: pandas dataframe
    Returns:
        words: list of lists of words
        words_labels: list of lists of labels
    '''
    words   = [sentence.split() for sentence in df['Sentence']]
    words_labels = [entity.split() for entity in df['Entity_sentence']]
    print("Number of sentences: ", len(words))
    print("Number of labels: ", len(words_labels))

    return words, words_labels


def MapLabels(words_labels, tag2idx):
    '''
    Function that maps the labels to the tags
    Args:
        words_labels: list of lists of labels
        tag2idx: dictionary that maps the labels to the tags
    Returns:
        labels: list of lists of tags
    '''
    labels = [[tag2idx.get(l) for l in lab] for lab in words_labels]
    return labels



def CreateDataset(words, words_labels, tag_values, IDS):
    '''
    Function that creates a dataset with id, words and labels
    Args:
        words: list of lists of words
        words_labels: list of lists of labels
        tag_values: list of the labels names
        IDS: list of the ids of the files
    Returns:
        dataset: dataset with id, words and labels
    '''
    features = Features({'id': Value(dtype='string', id=None), 'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 'ner_tags': Sequence(feature=ClassLabel(names=tag_values))})
    dataset = Dataset.from_dict({"id": IDS, "tokens": words, "ner_tags": words_labels}, features = features)
    return dataset