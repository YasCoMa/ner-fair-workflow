import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as classification_report_sk

from seqeval.metrics import accuracy_score, f1_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

def tokenize_and_align_labels(examples, tokenizer=None, flag_tokenizer=None, label_all_tokens=None):
    task = 'ner'

    '''
    Tokenize the sentences and align the labels with the tokens.

    This function takes a list of sentences and their corresponding labels, tokenizes the sentences, and aligns the labels with the tokens.
    The function handles special tokens by assigning them a label of -100, so they are automatically ignored in the loss function.
    The function also handles the case where a word is split into multiple tokens, assigning the label to the first token and either the same label or -100 to the other tokens, depending on the label_all_tokens flag.

    Args:
        examples (dict): A dictionary containing two keys:
            - "tokens": A list of sentences, where each sentence is a list of words.
            - "{task}_tags": A list of lists, each containing the labels for a sentence.

    Returns:
        tokenized_inputs (dict): A dictionary containing the tokenized sentences and their corresponding labels and word_ids.
            - "input_ids", "attention_mask", "token_type_ids": Lists of tokenized sentences.
            - "word_ids": A list of lists, each containing the word_ids for a sentence.
            - "labels": A list of lists, each containing the aligned labels for a sentence.
    '''

    if flag_tokenizer == 'LongFormer':
        tokenized_inputs = tokenizer(examples["tokens"], padding=True, truncation=True, max_length=1024, is_split_into_words=True)
    else:
        tokenized_inputs = tokenizer(examples["tokens"], padding=True, truncation=True, max_length=512, is_split_into_words=True)

    labels = []
    word_ids = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids.append(tokenized_inputs.word_ids(batch_index=i))
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids[-1]:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    
    tokenized_inputs['word_ids'] = word_ids
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def generate_true_predictions_and_labels(predictions, labels, label_list, mode = None):
    '''
    Generate true predictions and labels by removing special tokens.

    This function takes predictions and labels and filters out special tokens (with label -100).
    If mode is set to 'sklearn', it also filters out outside tokens (with label 0).
    It returns the true predictions and labels as lists of lists.

    Args:
        predictions (np.array): The predicted probabilities for each label. 
                            This is a 2D array where the first dimension is the number of examples and the second dimension is the number of possible labels.
        labels (list of list of int): The true labels for each example. 
                                  This is a list of lists where each inner list contains the labels for one example.
        label_list (list): A list of labels corresponding to the indices in the predictions.
        mode (str, optional): The mode to use for filtering labels. If set to 'sklearn', outside tokens (with label 0) are also filtered out.

    Returns:
        true_predictions (list of list): A list of lists, each containing the true predictions for a sentence.
        true_labels (list of list): A list of lists, each containing the true labels for a sentence.
    '''


    if mode == 'sklearn':
        # Remove ignored index (special tokens) and outside tokens
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100 and l!=0]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100 and l!=0]
            for prediction, label in zip(predictions, labels)
        ]
        
    else:
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

    return true_predictions, true_labels

def compute_objective(predictions, labels, label_list):
    '''
    Compute the F1 score between the true labels and the predicted labels.

    This function first converts the predictions to the labels using the argmax function.
    It then removes any special tokens that are represented by -100 in the labels.
    Finally, it computes and returns the F1 score between the true labels and the predicted labels.

    Args:
    predictions (np.array): The predicted probabilities for each label. 
                            This is a 2D array where the first dimension is the number of examples and the second dimension is the number of possible labels.
    labels (list of list of int): The true labels for each example. 
                                  This is a list of lists where each inner list contains the labels for one example.
    label_list (list): A list of labels corresponding to the indices in the predictions.

    Returns:
    float: The F1 score between the true labels and the predicted labels.
    '''

    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions, true_labels = generate_true_predictions_and_labels(predictions, labels, label_list)
    
    return f1_score(true_labels, true_predictions, zero_division=0)

def generate_csv_comparison(path_data, type_metrics = ['sk', 'strict', 'default'], type_level = ['token_level', 'word_level'], target_tags=[]):
    '''
    Generate csv files to compare the results reported with classification_report using different libraries (sklearn/seqeval)
    with the results reported in https://aclanthology.org/2022.wiesp-1.4.pdf

    This function reads the classification reports from csv files, calculates the mean, standard deviation, and maximum of the F1 scores,
    and compares these statistics with the results reported in the paper. The results are saved in a new csv file.

    Args:
        path_data (str): Path to the directory where the input csv files are located and where the output csv files will be saved.
        type_metrics (list of str): List of the types of metrics to be computed. The options are 'sk', 'strict', and 'lenient'. 
                                     The function will look for csv files that start with these strings in the path_data directory.

    Returns:
        None. The function saves the results in csv files in the path_data directory.
    '''
    for t in type_metrics: 
        for x in type_level:
            #Read the data
            print(f"Preparing csv comparison file... metric {t} - level {x}")
            files = [file for file in os.listdir(path_data) if re.match(f"^{t}_test_report_\d+.*_{x}\.csv$", file)]
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

            f1_scores = pd.DataFrame()
            for file in files:
                data = pd.read_csv(f"{path_data}/{file}", delimiter=',')
                f1_scores[file] = data['f1-score'].round(4)

            mean = f1_scores.mean(axis=1).round(4)
            sd = f1_scores.std(axis=1).round(4)
            max = f1_scores.max(axis=1).round(4)
        
            f1_scores['Support'] = data['support']
            f1_scores['Entity'] = data['Unnamed: 0']
            f1_scores.index = f1_scores['Entity']
            f1_scores = f1_scores.drop(columns=['Entity'])
        
            f1_scores['Mean'] = list(mean.astype(str) + ' (+-' + sd.astype(str) + ')')
            f1_scores['Just mean'] = list(mean)
            f1_scores['Max'] = list(max)
        
            #Put Support column the first one
            f1_scores = f1_scores[['Support'] + [col for col in f1_scores.columns if col != 'Support']]

            #Sort the index of the rows
            categories = list( map( lambda x: x[2:] if( x[:2].lower() in ['o-', 'b-', 'i-', 'e-']) else x, target_tags ) )
            aux = []
            for c in categories:
                if( c not in aux ):
                    aux.append(c)
            categories = aux
            if t != 'sk':
                f1_scores = f1_scores.reindex( categories + [ 'micro avg', 'macro avg', 'weighted avg'])
            else:
                f1_scores = f1_scores.reindex( categories + [ 'accuracy', 'macro avg', 'weighted avg'])
            
            f1_scores = f1_scores.drop(columns=['Just mean'])
            f1_scores.to_csv(f"{path_data}/f1_scores_{t}_{x}.csv", sep=',')

def make_confusion_matrix(cf,
                          path_save,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    It has been copied from https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py and adapted for saving the plots
    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    path_save:     Path to save the cm

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.2f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories, annot_kws={'size': 14})

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    
    #Added code
    plt.savefig(path_save)
    plt.close()

def generate_reports(predictions, labels, label_list, save_path, i, target_tags):
    '''
    Generate classification reports for the given predictions and labels.

    This function generates classification reports in lenient mode, strict mode with the IOB2 scheme, and sklearn mode.
    The reports are saved as CSV files in the specified save path. For lenient mode/sklearn mode the function computes
    the confusion matrix to have deeper error analysis at word level.

    Args:
        predictions (np.array): The predicted probabilities for each label. 
        labels (list of list of int): The true labels for each example.
        label_list (list): A list of labels corresponding to the indices in the predictions.
        save_path (str): The path where the report CSV files will be saved.
        i (int): The index of the current iteration.
        target_tags (list): Contains he entities contained in the experiment
    Returns:
        None
    '''
    # Remove ignored index (special tokens)
    true_predictions, true_labels = generate_true_predictions_and_labels(predictions, labels, label_list)

    #Compute the confusion matrix with prefixes
    aux_true_labels = [label for full_true in true_labels for label in full_true]
    aux_true_predictions = [label for full_pred in true_predictions for label in full_pred]

    #categories = ["O", "B-total-participants", "I-total-participants", "B-intervention-participants", "I-intervention-participants", "B-control-participants", "I-control-participants", "B-age", "I-age", "B-eligibility","I-eligibility","B-ethinicity","I-ethinicity","B-condition","I-condition","B-location","I-location","B-intervention","I-intervention","B-control","I-control","B-outcome","I-outcome","B-outcome-Measure","I-outcome-Measure","B-iv-bin-abs", "I-iv-bin-abs", "B-cv-bin-abs", "I-cv-bin-abs", "B-iv-bin-percent", "I-iv-bin-percent", "B-cv-bin-percent", "I-cv-bin-percent", "B-iv-cont-mean", "I-iv-cont-mean", "B-cv-cont-mean", "I-cv-cont-mean", "B-iv-cont-median", "I-iv-cont-median", "B-cv-cont-median", "I-cv-cont-median", "B-iv-cont-sd", "I-iv-cont-sd", "B-cv-cont-sd", "I-cv-cont-sd", "B-iv-cont-q1", "I-iv-cont-q1", "B-cv-cont-q1", "I-cv-cont-q1", "B-iv-cont-q3", "I-iv-cont-q3", "B-cv-cont-q3", "I-cv-cont-q3"]
    categories = target_tags
    cm = np.array(confusion_matrix(aux_true_labels, aux_true_predictions, labels=categories, normalize='true')).reshape(len(categories), len(categories))
    make_confusion_matrix(cm, f"{save_path}/cm_prefixes_{i}.png", categories=categories, cmap='viridis', figsize=(30,20), title='Confusion matrix', cbar = True, count = False, percent = False)

    #Compute the confusion matrix without prefixes
    aux_true_labels = [label.split('-',1)[1] if len(label.split('-')) > 1 else label for full_true in true_labels for label in full_true]
    aux_true_predictions = [label.split('-',1)[1] if len(label.split('-')) > 1 else label for full_pred in true_predictions for label in full_pred]
    
    #categories = ['O', 'total-participants', 'intervention-participants', 'control-participants', 'age', 'eligibility', 'ethinicity', 'condition', 'location', 'intervention', 'control', 'outcome', 'outcome-Measure', 'iv-bin-abs', 'cv-bin-abs', 'iv-bin-percent', 'cv-bin-percent', 'iv-cont-mean', 'cv-cont-mean', 'iv-cont-median', 'cv-cont-median', 'iv-cont-sd', 'cv-cont-sd', 'iv-cont-q1', 'cv-cont-q1', 'iv-cont-q3', 'cv-cont-q3']
    categories = list( map( lambda x: x[2:] if( x[:2].lower() in ['o-', 'b-', 'i-', 'e-']) else x, target_tags ) )
    aux = []
    for c in categories:
        if( c not in aux ):
            aux.append(c)
    categories = aux
    cm = np.array(confusion_matrix(aux_true_labels, aux_true_predictions, labels=categories, normalize='true')).reshape(len(categories), len(categories))
    make_confusion_matrix(cm, f"{save_path}/cm_{i}.png", categories=categories, cmap='viridis', figsize=(20,20), title='Confusion matrix', cbar = False, percent = False)

    # IO format
    class_report_lenient = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    pd.DataFrame(class_report_lenient).transpose().to_csv(f"{save_path}/default_test_report_{i}.csv")

    # IOB2 format
    class_report_strict = classification_report(true_labels, true_predictions, mode = 'strict', scheme=IOB2, output_dict=True, zero_division=0)
    pd.DataFrame(class_report_strict).transpose().to_csv(f"{save_path}/strict_test_report_{i}.csv")

    # Remove ignored index (special tokens) and outside tokens
    sk_true_predictions, sk_true_labels = generate_true_predictions_and_labels(predictions, labels, label_list, mode='sklearn')
    aux_true_labels = [label for full_true in sk_true_labels for label in full_true]
    aux_true_predictions = [label for full_pred in sk_true_predictions for label in full_pred]

    # Lenient mode
    class_report_sk = classification_report_sk(aux_true_labels, aux_true_predictions, output_dict=True, zero_division=0)
    pd.DataFrame(class_report_sk).transpose().to_csv(f"{save_path}/skp_test_report_{i}.csv")

    #Removing the prefixes
    aux_true_labels = [label.split('-',1)[1] if len(label.split('-')) > 1 else label for full_true in sk_true_labels for label in full_true]
    aux_true_predictions = [label.split('-',1)[1] if len(label.split('-')) > 1 else label for full_pred in sk_true_predictions for label in full_pred]

    class_report_sk = classification_report_sk(aux_true_labels, aux_true_predictions, output_dict=True, zero_division=0)
    pd.DataFrame(class_report_sk).transpose().to_csv(f"{save_path}/sk_test_report_{i}.csv")