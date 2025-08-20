import os
import re
import glob
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, cohen_kappa_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import accuracy_score as accuracy_score_sk, classification_report as classification_report_sk

from seqeval.scheme import IOB2
from seqeval.metrics import accuracy_score, f1_score, classification_report

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

def __get_probabilities( labels, target_tags, outputs_logits, labels_to_ignore=[-100, 0], remove_prefix=False):
    auxc = __remove_prefix_tags(target_tags)

    arr = torch.softmax( outputs_logits, dim=2).to('cpu').numpy()
    probs = []
    for idx, sentence_labels in enumerate(labels):
        for lidx, l in enumerate(sentence_labels):
            if(l not in labels_to_ignore):
                px = arr[idx][lidx]

                if(remove_prefix):
                    rep = {}
                    for c, p in zip( auxc, px ):
                        if(not c in rep):
                            rep[c] = []
                        rep[c].append(p)
                    
                    px = [ max(rep[c]) for c in rep ]
                probs.append( px )
    return probs

def __get_binary_truey(labels, target_tags, labels_to_ignore=[-100, 0], remove_prefix=False):
    y = []
    for idx, sentence_labels in enumerate(labels):
        for lidx, l in enumerate(sentence_labels):
            if(l not in labels_to_ignore):
                v = [ target_tags[l] ]
                if(remove_prefix):
                    v = __remove_prefix_tags(v)
                y.append( v )

    categories = target_tags
    if(remove_prefix):
        categories = __remove_prefix_tags(target_tags, remove_duplicates=True)
    yaux = [ [l] for l in categories ]
    enc = OneHotEncoder()
    enc.fit(yaux)
    ency = enc.transform(y).toarray()

    return ency, categories

def __remove_prefix_tags(target_tags, remove_duplicates=False):
    categories = list( map( lambda x: x[2:] if( x[:2].lower() in ['o-', 'b-', 'i-', 'e-', 's-', 'u-', 'l-']) else x, target_tags ) )
    
    if( remove_duplicates ):
        aux = []
        for j, c in enumerate(categories):
            if( c not in aux ):
                aux.append(c)
        categories = aux
    return categories

def __flatten_array(predictions, labels, remove_prefix=False):
    arr1dpred = []
    arr1dy = []
    for i, vpr in enumerate(predictions):
        arr1dpred.extend( vpr )
        arr1dy.extend( labels[i] )
    
    if(remove_prefix):
        arr1dpred = __remove_prefix_tags(arr1dpred)
        arr1dy = __remove_prefix_tags(arr1dy)
    return arr1dpred, arr1dy

def _rename_label_predictions(predictions, labels, target_tags, labels_to_ignore=[-100, 0]):
    
    y = []
    preds = []
    for idx, sentence_labels in enumerate(labels):
        auxy = []
        auxp = []
        mlen = len(predictions[idx])
        
        for lidx, l in enumerate(sentence_labels):
            if( (lidx < mlen) and (l not in labels_to_ignore) ):
                auxy.append( target_tags[l] )
                auxp.append( target_tags[ predictions[idx][lidx] ] )
        y.append( auxy )
        preds.append( auxp )
        
    return preds, y

def _generate_metric_plots(dat, target_tags, report_identifier, out_path):
    for k in dat:
        categories = target_tags
        if( k == 'without-prefix' ):
            categories = __remove_prefix_tags(target_tags, remove_duplicates=True)

        cm = np.array( confusion_matrix( dat[k][1], dat[k][0], labels=categories, normalize='true')).reshape(len(categories), len(categories) )
        make_confusion_matrix(cm, f"{out_path}/cm_{k}_{report_identifier}.png", categories=categories, cmap='viridis', figsize=(30,20), title='Confusion matrix', cbar = True, count = False, percent = False)

def __compute_aucroc(labels, outputs_logits, target_tags, remove_prefix):
    probs = __get_probabilities( labels, target_tags, outputs_logits, labels_to_ignore=[-100, 0], remove_prefix=remove_prefix)
    onehot_y, categories = __get_binary_truey(labels, target_tags, labels_to_ignore=[-100, 0], remove_prefix=remove_prefix)
    
    auc_roc = roc_auc_score(onehot_y, probs, average=None)
    per_class = {}
    for i, c in enumerate(categories):
        per_class[c] = auc_roc[i]
        if( str(per_class[c]) == 'nan' ):
            per_class[c] = 0
    return per_class

def _generate_summary_metrics_bysklearn(dat, labels, outputs_logits, target_tags, report_identifier, level, out_path):
    index = report_identifier.split('@')[-1]

    categories = target_tags
    for k in dat:
        remove_prefix = False
        if( k == 'without-prefix' ):
            remove_prefix = True

        auc_roc = 0
        if(outputs_logits != None):
            auc_roc = __compute_aucroc(labels, outputs_logits, target_tags, remove_prefix)
        mcc = matthews_corrcoef( dat[k][1], dat[k][0] )
        kappa = cohen_kappa_score( dat[k][1], dat[k][0] )
        acc = accuracy_score_sk( dat[k][1], dat[k][0] )
        class_report_sk = classification_report_sk( dat[k][1], dat[k][0], output_dict=True, zero_division=0)
        
        df = pd.DataFrame(class_report_sk).transpose()
        
        ordered_auc = 0
        if(outputs_logits != None):
            ordered_auc = []
            for i in df.index:
                v = 0
                if( i in auc_roc ):
                    v = auc_roc[i]
                ordered_auc.append( v )
        df['aucroc'] = ordered_auc

        df['mcc'] = mcc
        df['kappa'] = kappa
        df['accuracy'] = acc

        df.to_csv(f"{out_path}/sk-{k}_report_{report_identifier}-l{level}.tsv", sep='\t')

def _generate_summary_metrics_byseqeval(slabels, spredictions, report_identifier, level, out_path):
    index = report_identifier.split('@')[-1]

    acc = accuracy_score( slabels, spredictions )
    modes = { 'default': None, 'strict': 'strict' }
    for m in modes:
        class_report = classification_report( slabels, spredictions, mode = modes[m], scheme=IOB2, output_dict=True, zero_division=0)
        
        df = pd.DataFrame(class_report).transpose()
        df['accuracy'] = acc

        df.to_csv(f"{out_path}/seqeval-{m}_report_{report_identifier}-l{level}.tsv", sep='\t')

def generate_reports_table( outputs_logits, predictions, labels, target_tags, out_path, report_identifier, index='unique', level='token' ):
    '''
    Generate reports considering the modes available in seqeval (IO (default) or IOB (strict) formats) and sklearn (with or without prefixes).

    Args:
        outputs (tensor): A torch tensor containing the instances, the labels predicted for each token of the instance and the probabilities of each label for a single token
        predictions (A list of lists): contains the list of labels predicted for each token for the instances
        labels (A list of lists): contains the list of real labels for each token
        target_tags (list): the list of possible tags that belong to the NER experiment
        out_path (string): path to the directory where the reports and plots will be saved
        report_identifier (string): an identifier for the report
        index (string): In case it is a report of model replicates (training_0, training_1...) this index is the number itself, if it is not a sequence, the default is 'unique'
        level (string): Two levels are accepted (token or word)
    '''

    spredictions, slabels = _rename_label_predictions(predictions, labels, target_tags, labels_to_ignore=[-100, 0])
    
    dat = { 'with-prefix': '', 'without-prefix': '' }
    dat['with-prefix'] = __flatten_array(spredictions, slabels, remove_prefix=False)
    dat['without-prefix'] = __flatten_array(spredictions, slabels, remove_prefix=True)
    
    report_identifier = f"{report_identifier}@{index}"
    _generate_summary_metrics_bysklearn(dat, labels, outputs_logits, target_tags, report_identifier, level, out_path)
    _generate_summary_metrics_byseqeval(slabels, spredictions, report_identifier, level, out_path)

    _generate_metric_plots(dat, target_tags, report_identifier, out_path)

def _generate_summary_plot(inpath, out_path, fname, agg_stats_metric = 'median'):
    stats_metric = agg_stats_metric.capitalize().replace('_', ' ').replace('Std', 'Standard Deviation')

    df = pd.read_csv( inpath, sep='\t', index_col=0)
    f = df[ (df['evaluation_metric'] != 'support') & (df['stats_agg_name'] == 'median') ].reset_index()
    f = f[ ['Entity', 'evaluation_metric', 'stats_agg_value'] ]
    f.columns = ['Entity', 'Evaluation Metric', stats_metric ]
    fig = px.bar(f, x="Entity", y=stats_metric, color="Evaluation Metric", barmode="group")
    path = os.path.join(out_path, fname)
    fig.write_image(path)

def aggregate_reports(entry_point, report_identifier, out_path, agg_stats_metric = 'median', levels = ['word', 'token']):
    evaluation_modes = ['seqeval-default', 'seqeval-strict', 'sk-with-prefix', 'sk-without-prefix']
    
    for mode in evaluation_modes:
        for level in levels:
            inpath = os.path.join(entry_point, f"{mode}_report_{report_identifier}@*-l{level}.tsv")
            files = glob.glob( inpath )
            if( len( files ) > 0 ):
                file_names = []
                dat = {}
                for f in files:
                    fname = f.split('/')[-1].split('.')[0]
                    file_names.append(fname)
                    df = pd.read_csv( f, sep='\t', index_col=0 )
                    for m in df.columns:
                        for idx in df.index:
                            if( (idx.find(' avg') == -1) and (idx!='accuracy') and (idx!='O') ):
                                if( not idx in dat ):
                                    dat[idx] = {}
                                if( not m in dat[idx] ):
                                    dat[idx][m] = { 'values': [] }
                                dat[idx][m]['values'].append(df.loc[idx, m])

                dtst = {}
                for tag in dat:
                    for m in dat[tag]:
                        vs = dat[tag][m]['values']
                        dat[tag][m]['mean'] = np.average(vs)
                        dat[tag][m]['std'] = np.std(vs)
                        dat[tag][m]['median'] = np.median(vs)
                        dat[tag][m]['max'] = np.max(vs)
                        dat[tag][m]['min'] = np.min(vs)

                header_values = '\t'.join( [ f"metric_value-{fname}" for fname in file_names ] )
                fname = f"{mode}_summary-report_{report_identifier}-l{level}.tsv"
                opath = os.path.join(out_path, fname)
                f = open( opath, 'w')
                f.write( f"Entity\tevaluation_metric\tstats_agg_name\tstats_agg_value\t{header_values}\n")
                for tag in dat:
                    for m in dat[tag]:
                        values = '\t'.join([ str(x) for x in dat[tag][m]['values'] ])
                        for stm in dat[tag][m]:
                            if(stm != 'values'):
                                v = dat[tag][m][stm]
                                f.write( f"{tag}\t{m}\t{stm}\t{v}\t{values}\n" )
                f.close()

                fname = f"plot_{mode}_summary-report_{report_identifier}-l{level}.png"
                _generate_summary_plot(opath, out_path, fname, agg_stats_metric = 'median')

def __compute_objective(predictions, labels, label_list, metric='f1'):
    '''
    Compute the chosen metric score between the true labels and the predicted labels.

    This function first converts the predictions to the labels using the argmax function.
    It then removes any special tokens that are represented by -100 in the labels.
    Finally, it computes and returns the F1 score between the true labels and the predicted labels.

    Args:
    predictions (np.array): The predicted probabilities for each label. 
                            This is a 2D array where the first dimension is the number of examples and the second dimension is the number of possible labels.
    labels (list of list of int): The true labels for each example. 
                                  This is a list of lists where each inner list contains the labels for one example.
    label_list (list): A list of labels corresponding to the indices in the predictions.

    metric (string): The metric chosen for optimization (f1, accuracy, precision or recall)

    Returns:
    float: The metric score between the true labels and the predicted labels.
    '''

    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions, true_labels = _rename_label_predictions(predictions, labels, label_list)
    
    return eval(f'{metric}_score')(true_labels, true_predictions, zero_division=0)

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
    true_predictions, true_labels = _rename_label_predictions(predictions, labels, label_list, labels_to_ignore=[-100, 0])

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
    categories = __remove_prefix_tags(target_tags, remove_duplicates=True)
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