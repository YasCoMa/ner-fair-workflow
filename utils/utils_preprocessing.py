import os
from datasets import Dataset, Features, Value, Sequence, ClassLabel
import pandas as pd

'''
Script with functions to preprocess the data and analyze the mismatch between the data and the .ann files
Author: Carlos Cuevas Villarmin
Last update: 31/05/2024
'''

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