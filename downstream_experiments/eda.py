import os
import re
import sys
import pickle
import optuna
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from scipy.stats import ranksums
from scipy import stats

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class ExplorationBenchmarkDss:
    def __init__(self, fout):
        self.out = os.path.join(fout, 'eda_paper_material')
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )

    # -------- General
    def _write_latex_table(self, opath, lines):
        ncols = len( lines[0].split(' & ') )

        f = open( opath, 'w')
        f.write("""
\\begin{table}[]
\\begin{tabular}{%s}
            """ %( ('l')*ncols ) )
        f.write( '\n'.join(lines)+'\\\\\n' )
        f.write("""
\\end{tabular}
\\end{table}
            """)
        f.close()


    def wrap_picods_comp_metrics_reprod(self):
        # PICO reproducibility
        ## Generate table with hyperparameters
        hprefs = ["Experiment-1", "Experiment-2", "nerfairwf"]
        hyperparams = {
            "Learning rate": dict(zip( hprefs, ["2e-5", "4.976e-5", "4.053e-5"] )),
            "Weigth decay": dict(zip( hprefs, [ "1e-2", "3e-3", "1.013e-4"] )),
            "per device train batch size": dict(zip( hprefs, ["16", "8", "8"] )),
            "per device eval batch size": dict(zip( hprefs, ["16", "16", "16"] ))
        }
        lines = [ ['Parameter/Experiment'] + hprefs ]
        lines[0] = [ f"\\textbf{{ {v} }}" for v in lines[0] ]
        for k in hyperparams:
            lines.append( [k] + list( hyperparams[k].values() ) )
        lines = list( map( lambda x: ' & '.join(x)+'\\\\', lines ))

        opath = os.path.join(self.out, "table_latex_hyperparams.txt")
        self._write_latex_table(opath, lines)

        ## Generate comparison table of f1-scores
        sota = """Entity PICO-Reference Experiment-1 Experiment-2
total-participants 0.94 0.9065(+-0.0096) 0.9313(+-0.0048)
intervention-participants 0.85 0.7431(+-0.0123) 0.8177(+-0.0135)
control-participants 0.88 0.7846(+-0.0108) 0.8480(+-0.0124)
age 0.80 0.5638(+-0.0300) 0.5724(+-0.0731)
eligibility 0.74 0.6049(+-0.0131) 0.6382(+-0.0219)
ethinicity 0.88 0.7135(+-0.0433) 0.7163(+-0.0353)
condition 0.80 0.6412(+-0.0469) 0.7122(+-0.0421)
location 0.76 0.6156(+-0.0226) 0.6258(+-0.0363)
intervention 0.84 0.7805(+-0.0047) 0.7899(+-0.0095)
control 0.76 0.6780(+-0.0205) 0.6529(+-0.0190)
outcome 0.81 0.6321(+-0.0056) 0.6667(+-0.0151)
outcome-Measure 0.84 0.7441(+-0.0274) 0.8003(+-0.0240)
iv-bin-abs 0.80 0.6184(+-0.0278) 0.7640(+-0.0352)
cv-bin-abs 0.82 0.6557(+-0.0214) 0.8195(+-0.0219)
iv-bin-percent 0.87 0.6460(+-0.0174) 0.6731(+-0.0317)
cv-bin-percent 0.88 0.6919(+-0.0224) 0.7549(+-0.0233)
iv-cont-mean 0.81 0.5081(+-0.0352) 0.4271(+-0.0334)
cv-cont-mean 0.86 0.4711(+-0.0160) 0.4117(+-0.0297)
iv-cont-median 0.75 0.6630(+-0.0336) 0.7415(+-0.0216)
cv-cont-median 0.79 0.6937(+-0.0195) 0.7769(+-0.0373)
iv-cont-sd 0.83 0.4606(+-0.0424) 0.6274(+-0.0683)
cv-cont-sd 0.82 0.4711(+-0.0514) 0.7264(+-0.0826)
iv-cont-q1 0 0.0000(+-0.0000) *
cv-cont-q1 0 0.0000(+-0.0000) *
iv-cont-q3 0 0.0000(+-0.0000) *
cv-cont-q3 0 0.0000(+-0.0000) *
micro-avg - 0.6845(+-0.0032) 0.7261(+-0.0119)
macro-avg 0.6973 0.5495(+-0.0022) 0.7043(+-0.0138)
weighted-avg 0.8282 0.6872(+-0.0031) 0.7273(+-0.0118)"""
        dat = {}
        lines = sota.split('\n')
        exps = lines[0].split(' ')[1:]
        for l in lines[1:]:
            els = l.split(' ')
            dat[ els[0] ] = { "nerfairwf": {} }
            for i,v in enumerate(els[1:]):
                parts = v.split('(+-')
                dat[ els[0] ][ exps[i] ] = { 'mean': parts[0], 'std': '' }
                if( len(parts) > 1 ):
                    dat[ els[0] ][ exps[i] ]['std'] = parts[1].replace(')','')

        mode = "sk-without-prefix"
        level = "ltoken"
        m = "f1-score"
        indir = '/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/biobert-original-hypersearch-biobert-base-cased-v1.2-finetuned-ner/'
        pathdir = os.path.join( indir, 'test', 'summary_reports' )
        fname = f"{mode}_summary-report_test-model-{level}.tsv"
        path = os.path.join(pathdir, fname)
        df = pd.read_csv( path, sep= '\t')
        aux = df[ (df['evaluation_metric'] == m) & (df['stats_agg_name'].isin(['mean','std']) ) ]
        for i in aux.index:
            entity = df.loc[i,'Entity']
            aggf = df.loc[i, 'stats_agg_name']
            aggv = df.loc[i, 'stats_agg_value']
            dat[entity]['nerfairwf'][aggf] = aggv

        exps += ['nerfairwf']
        lines = [ ['Entity/Experiment'] + exps ]
        lines[0] = [ f"\\textbf{{ {v} }}" for v in lines[0] ]
        for e in dat:
            try:
                l = [e]
                for k in exps:
                    mean = dat[e][k]['mean']
                    std = dat[e][k]['std']
                    if( std != ''):
                        v = "%.4f (+- %.4f)" %( float(mean), float(std) )
                    else:
                        if( mean.isnumeric() ):
                            v = "%.4f" %( float(mean) )
                        else:
                            v = "%s" %( mean )
                    l.append( v )
                lines.append( l )
            except:
                pass
        lines = list( map( lambda x: ' & '.join(x)+'\\\\', lines ))

        opath = os.path.join(self.out, "table_latex_pico_metrics.txt")
        self._write_latex_table(opath, lines)

    def wrap_bench_dss_eval_metrics_reprod(self):
        evaluation_modes = ['seqeval-default', 'seqeval-strict', 'sk-with-prefix', 'sk-without-prefix']
        levels = ['ltoken', 'lword']
        target_metrics = ['mcc', 'f1-score']

        indir = '/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/trials/biobert-__db__-hypersearch-biobert-base-cased-v1.2-finetuned-ner'
        dss = [ "bc5cdr", "ncbi", "biored", "chiads", "merged_train"]
        
        # Choose mode eval with best metric values (f1 and mcc)
        best = {}
        for ds in dss:
            pathdir = os.path.join( indir.replace('__db__', ds), 'test', 'summary_reports' )
            for level in levels:
                key = f"{ds}#{level}"
                if(not key in best):
                    best[key] = {}

                for m in target_metrics:
                    dat = {}
                    for mode in evaluation_modes:
                        fname = f"{mode}_summary-report_test-model-{level}.tsv"
                        path = os.path.join(pathdir, fname)
                        if( not os.path.exists(path) ):
                            fname = f"{mode}_summary-report_merged-test-{level}.tsv"
                            path = os.path.join(pathdir, fname)

                        df = pd.read_csv( path, sep= '\t')
                        aux = df[ (df['evaluation_metric'] == m) & (df['stats_agg_name'] == 'mean') ]
                        if( len(aux) > 0 ):
                            dat[mode] = aux.stats_agg_value.max()
                    odat = dict( sorted( dat.items(), key=lambda item: item[1], reverse=True ) )
                    best[key][m] = [ list(odat)[0], list(odat.values())[0] ]
                    print(key, m, best[key][m])

        lines = []
        l = ["Dataset", "Entity", "Level", "Metric", "Value"]
        l = '\t'.join( [ str(x) for x in l] )
        lines.append(l)
        for k in best:
            ds, level = k.split('#')

            pathdir = os.path.join( indir.replace('__db__', ds), 'test', 'summary_reports' )
            for m in best[k]:
                mode = best[key][m]
                #print(ds, level, m, mode)
                mode = mode[0]

                fname = f"{mode}_summary-report_test-model-{level}.tsv"
                path = os.path.join(pathdir, fname)
                if( not os.path.exists(path) ):
                    fname = f"{mode}_summary-report_merged-test-{level}.tsv"
                    path = os.path.join(pathdir, fname)

                df = pd.read_csv( path, sep= '\t')
                ents = df[ (df['evaluation_metric'] == m) & (df['stats_agg_name'] == 'mean') ].Entity.unique()
                aux = df[ (df['evaluation_metric'] == m) & (df['stats_agg_name'] == 'mean') & (df['Entity'].isin(ents)) ]
                for i in aux.index:
                    entity = df.iloc[i, 0]
                    values = df.iloc[i, 4:]
                    for v in values:
                        l = [ ds, entity, level, m, v ]
                        l = '\t'.join( [ str(x) for x in l] )
                        lines.append(l)
        opath = os.path.join(self.out, "table_comp_sota.tsv")
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()

        m = 'f1-score'
        df = pd.read_csv( opath, sep= '\t')
        for m in target_metrics:
            for ds in dss:
                cutoff = 0.5
                if( m == 'mcc' ):
                    cutoff = 0
                aux = df[ (df["Dataset"] == ds) & (df["Metric"] == m) & (df['Value'] > cutoff) ]
                aux.columns = ["Dataset", "Entity", "Level", "Metric", m]
                fig = px.box( aux, x="Entity", y=m, color="Level")
                fig.update_layout( title_text = "Dataset "+ds, yaxis_range = [0, 1] )
                opath = os.path.join(self.out, f'{ds}_{m}_comparison_sota.png')
                fig.write_image( opath )

    def _count_annotations(self, pathdir, files):
        dat = {}
        total_classes = 0
        for f in files:
            path = os.path.join( pathdir, f )
            pmid = f.split('.')[0]
            f = open(path, 'r')
            for line in f:
                l = line.split('\t')
                entity = l[1].split(' ')[0]
                uid = l[0]
                annkey = f"{pmid}_{uid}"
                if( not entity in dat ):
                    dat[entity] = { 'pmids': set(), 'annotations': set() }
                dat[entity]['pmids'].add(pmid)
                dat[entity]['annotations'].add(annkey)
            f.close()

            total_classes = len(dat)

        return dat, total_classes

    def gen_suppTable_counts_annotations(self):
        dat = {}

        # ---------- PICO gold dataset
        indir = '/aloy/home/ymartins/match_clinical_trial/experiments/data'
        ds = 'pico_gold'
        pathdir = indir
        files = list( filter( lambda x: x.endswith('.ann'), os.listdir(pathdir) ) )
            
        dt, total_classes = self._count_annotations( pathdir, files)
        dat[ds] = {}
        dat[ds]['details'] = dt
        dat[ds]['count_classes'] = total_classes
        
        # ---------- Benchmark datasets
        indir = '/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/'
        dss = [ "bc5cdr", "ncbi", "biored", "chiads", "merged_train"]
        for ds in tqdm(dss):
            dat[ds] = {}

            pathdir = os.path.join(indir, 'nerdata', ds, 'processed')
            if( ds.startswith("merged") ):
                pathdir = os.path.join(indir, ds)
            files = list( filter( lambda x: x.endswith('.ann'), os.listdir(pathdir) ) )
            
            dt, total_classes = self._count_annotations( pathdir, files)
            dat[ds]['details'] = dt
            dat[ds]['count_classes'] = total_classes
        
        # Parsing and writing information
        lines = []
        header = ["dataset", "entity", "number_entities", "number_papers", "number_annotations"]
        lines.append( '\t'.join(header) )
        for ds in dat:
            total = dat[ds]['count_classes']
            info = dat[ds]['details']
            for e in info:
                cnt_pub = len( info[e]['pmids'] )
                cnt_ann = len( info[e]['annotations'] )
                l = '\t'.join( [ ds, e, str(total), str(cnt_pub), str(cnt_ann) ] )
                lines.append(l)
        
        opath = os.path.join( self.out, 'datasets_meta_metrics.tsv')
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()

        ltines = list( map( lambda x: ' & '.join( x.split('\t') )+'\\\\', lines ))
        opath = os.path.join(self.out, "table_latex_dss_meta_metrics.txt")
        self._write_latex_table(opath, ltines)

    def _check_correlation_count_eval(self):
        entities = set()
        cp = []
        opath = os.path.join(self.out, "table_latex_pico_metrics.txt")
        f = open( opath, 'r')
        for line in f:
            if(line.find('+-') != -1):
                l = line.split(' & ')
                ent = l[0]
                entities.add(ent)
                cp.append( float(l[-1].split(' ')[0]) )
        f.close()

        opath = os.path.join( self.out, 'datasets_meta_metrics.tsv')
        df = pd.read_csv( opath, sep='\t')
        df = df[ ( df['dataset'] == 'pico_gold' ) & ( df['entity'].isin(entities) ) ]
        entities = df.entity.unique()
        cpaper = df.number_papers.values.tolist()
        canns = df.number_annotations.values.tolist()

        # Papers x f1
        res = stats.pearsonr(cpaper, cp)
        print("Papers - Corr.:", res.statistic, ' P-value:', res.pvalue)

        # Annotations x f1
        res = stats.pearsonr(canns, cp)
        print("Annotations - Corr.:", res.statistic, ' P-value:', res.pvalue)

    def check_best_merged_individual_models(self):
        dat = {}
        
        inpath = os.path.join( self.out, 'table_comp_sota.tsv')
        df = pd.read_csv( inpath, sep='\t')
        merg_ents = df[ df['Dataset'] == 'merged_train' ].Entity.unique()

        # organize data
        for i in df.index:
            ent = df.loc[i, 'Entity']
            lv = df.loc[i, 'Level']
            mt = df.loc[i, 'Metric']
            key = f"{ent}#{lv}#{mt}"
            if( not key in dat):
                    dat[key] = {}

            ds = df.loc[i, 'Dataset']
            if( not ds in dat[key]):
                dat[key][ds] = []

            v = df.loc[i, 'Value']
            dat[key][ds].append(v)

        # process and compute best for each entity, metirc and level
        lines = []
        header =  ["Entity", "Level", "Metric", "Dataset", "Mean Dataset", "Mean Merged", "Best", "p-value", "diff (merged - ds)"]
        lines.append( '\t'.join(header) )
        for k in dat:
            ent, level, metric = k.split('#')
            arrm = dat[k]["merged_train"]
            mean_merg = sum(arrm)/len(arrm)

            for ds in dat[k]:
                if( ds != "merged_train" ):
                    arrd = dat[k][ds]
                    mean_ds = sum(arrd)/len(arrd)
                    best = 'merged_train'
                    if( mean_ds > mean_merg ):
                        best = 'individual'
                    pval = ranksums(arrm, arrd).pvalue
                    diff = mean_merg - mean_ds
                    l = [ent, level, metric, ds, mean_ds, mean_merg, best, pval, diff]
                    l = '\t'.join( [ str(v) for v in l ] )
                    lines.append(l)

        # export result
        opath = os.path.join( self.out, 'table_comp_summary_mergedxInd_sota.tsv')
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()

        # exploring it:
        '''
        df = pd.read_csv('outnerwf/eda_paper_material/table_comp_summary_mergedxInd_sota.tsv', sep='\t')
        # Datasets positively affected by merged model:
        dss = df[ df['Best']=='merged_train' ].Dataset.unique()

        # Entities positively affected by merged model:
        df[ df['Best']=='merged_train' ].Entity.unique()

        # Counts of entities grouped by level, metric, dataset
        cgrouped = df[ df['Best']=='merged_train' ].groupby(['Level','Metric','Dataset']).count()
        
        # Getting expressive differences:
        df['diff'] = df['Mean Merged'] - df['Mean Dataset']
        df_highest_diffs = df[ df['Best']=='merged_train' ].sort_values(by='diff', ascending=False).head(30)
        '''

    def test_significance_module(self):
        cpath = '/aloy/home/ymartins/match_clinical_trial/experiments/config_biobert_hypersearch.json'
        cnf = json.load( open( cpath, 'r') )

        if( not 'external_eval_data' in cnf ):
            ## Generate comparison table of f1-scores
            sota = """Entity PICO-Reference Experiment-1 Experiment-2
    total-participants 0.94 0.9065(+-0.0096) 0.9313(+-0.0048)
    intervention-participants 0.85 0.7431(+-0.0123) 0.8177(+-0.0135)
    control-participants 0.88 0.7846(+-0.0108) 0.8480(+-0.0124)
    age 0.80 0.5638(+-0.0300) 0.5724(+-0.0731)
    eligibility 0.74 0.6049(+-0.0131) 0.6382(+-0.0219)
    ethinicity 0.88 0.7135(+-0.0433) 0.7163(+-0.0353)
    condition 0.80 0.6412(+-0.0469) 0.7122(+-0.0421)
    location 0.76 0.6156(+-0.0226) 0.6258(+-0.0363)
    intervention 0.84 0.7805(+-0.0047) 0.7899(+-0.0095)
    control 0.76 0.6780(+-0.0205) 0.6529(+-0.0190)
    outcome 0.81 0.6321(+-0.0056) 0.6667(+-0.0151)
    outcome-Measure 0.84 0.7441(+-0.0274) 0.8003(+-0.0240)
    iv-bin-abs 0.80 0.6184(+-0.0278) 0.7640(+-0.0352)
    cv-bin-abs 0.82 0.6557(+-0.0214) 0.8195(+-0.0219)
    iv-bin-percent 0.87 0.6460(+-0.0174) 0.6731(+-0.0317)
    cv-bin-percent 0.88 0.6919(+-0.0224) 0.7549(+-0.0233)
    iv-cont-mean 0.81 0.5081(+-0.0352) 0.4271(+-0.0334)
    cv-cont-mean 0.86 0.4711(+-0.0160) 0.4117(+-0.0297)
    iv-cont-median 0.75 0.6630(+-0.0336) 0.7415(+-0.0216)
    cv-cont-median 0.79 0.6937(+-0.0195) 0.7769(+-0.0373)
    iv-cont-sd 0.83 0.4606(+-0.0424) 0.6274(+-0.0683)
    cv-cont-sd 0.82 0.4711(+-0.0514) 0.7264(+-0.0826)
    iv-cont-q1 0 0.0000(+-0.0000) *
    cv-cont-q1 0 0.0000(+-0.0000) *
    iv-cont-q3 0 0.0000(+-0.0000) *
    cv-cont-q3 0 0.0000(+-0.0000) *
    micro-avg - 0.6845(+-0.0032) 0.7261(+-0.0119)
    macro-avg 0.6973 0.5495(+-0.0022) 0.7043(+-0.0138)
    weighted-avg 0.8282 0.6872(+-0.0031) 0.7273(+-0.0118)"""
            dat = {}
            lines = sota.split('\n')
            exps = lines[0].split(' ')[1:]
            for i,e in enumerate(exps):
                dat[ exps[i] ] = {}
            for l in lines[1:]:
                els = l.split(' ')
                for i,v in enumerate(els[1:]):
                    parts = v.split('(+-')
                    dat[ exps[i] ][ els[0] ] = { 'mean': parts[0], 'std': '' }
                    if( len(parts) > 1 ):
                        dat[ exps[i] ][ els[0] ]['std'] = parts[1].replace(')','')

            cnf["external_eval_data"] = { 
                "mode": "global",
                "agg_function": "mean",
                "evaluators": [ ] 
            }
            for ev in dat:
                obj = { "identifier": ev, "level": "token", "results": {} }
                for e in dat[ev]:
                    obj["results"][e]["f1-score"] = float( dat[ev][e]['mean'] )
                    cnf["external_eval_data"]["evaluators"].append(obj)
            json.dump(cnf, open(cpath, 'w') )

        l = "/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/"
        os.system( "rm %s/logs/*significance_analysis.ready" %(l) )

        cmd = f"nextflow run /aloy/home/ymartins/match_clinical_trial/ner_subproj/main.nf --dataDir /aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/ --runningConfig {cpath} --mode 'significance_analysis'"
        os.system(cmd)

    def run(self):
        #self.wrap_picods_comp_metrics_reprod()
        #self.wrap_bench_dss_eval_metrics_reprod()
        #self.gen_suppTable_counts_annotations()
        #self._check_correlation_count_eval()

        #self.check_best_merged_individual_models()
        self.test_significance_module()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/outnerwf'
    i = ExplorationBenchmarkDss( odir )
    i.run()