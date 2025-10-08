import os
import re
import sys
import pickle
import optuna
import pandas as pd
import plotly.express as px
from scipy.stats import ranksums

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class ExplorationBenchmarkDss:
    def __init__(self, fout):
        self.out = os.path.join(fout, 'eda_paper_material')
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
            
        self.out_ct_processed = os.path.join( fout, "processed_cts" )

    # -------- General
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
        lines[0] = [ f"\\textbf\{ {v} \}" for v in lines[0] ]
        for k in hyperparams:
            lines.append( [k] + list( hyperparams[k].values() ) )
        lines = list( map( lambda x: ' & '.join(x), lines ))

        opath = os.path.join(self.out, "table_latex_hyperparams.txt")
        f = open( opath, 'w')
        f.write("""
\\begin\{table\}[]
\\begin\{tabular\}\{llll\}
            """)
        f.write( '\n'.join(lines)+'\\\\\n' )
        f.write("""
\\end\{tabular\}
\\end\{table\}
            """)
        f.close()

        ## Generate comparison table of f1-scores
        sota = """Entity Ref[6] Experiment-1 Experiment-2
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
micro avg - 0.6845(+-0.0032) 0.7261(+-0.0119)
macro avg 0.6973 0.5495(+-0.0022) 0.7043(+-0.0138)
weighted avg 0.8282 0.6872(+-0.0031) 0.7273(+-0.0118)"""
        dat = {}
        lines = sota.split('\n')
        exps = lines[0].split(' ')[1:]
        for l in lines[1:]:
            els = l.split(' ')
            dat[ els[0] ] = { "nerfairwf": {} }
            for i,v in enumerate(els[1:]):
                parts = v.split('(+-')
                dat[ els[0] ][ exps[i] ]['mean'] = parts[0]
                dat[ els[0] ][ exps[i] ]['std'] = ""
                if( len(parts) > 1 ):
                    dat[ els[0] ][ exps[i] ]['std'] = parts[1]

        mode = "sk-without-prefix"
        level = "ltoken"
        m = "f1-score"
        indir = '/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/biobert-original-hypersearch-biobert-base-cased-v1.2-finetuned-ner/test/'
        pathdir = os.path.join( indir, 'test', 'summary_reports' )
        fname = f"{mode}_summary-report_test-model-{level}.tsv"
        path = os.path.join(pathdir, fname)
        df = pd.read_csv( path, sep= '\t')
        aux = df[ (df['evaluation_metric'] == m) & (df['stats_agg_name'].isin(['mean','std']) ) ]
        for i in df.index:
            entity = df.loc[i,'Entity']
            aggf = df.loc[i, 'stats_agg_name']
            aggv = df.loc[i, 'stats_agg_value']
            dat[entity]['nerfairwf'][aggf] = aggv

        exps += ['nerfairwf']
        lines = [ ['Entity/Experiment'] + exps ]
        lines[0] = [ f"\\textbf\{ {v} \}" for v in lines[0] ]
        for e in dat:
            l = [e]
            for k in exps:
                mean = dat[e][k]['mean']
                std = dat[e][k]['std']
                v = f"{mean} (+- {std})"
                l.append( v )
            lines.append( l )

        lines = list( map( lambda x: ' & '.join(x), lines ))

        opath = os.path.join(self.out, "table_latex_pico_metrics.txt")
        f = open( opath, 'w')
        f.write("""
\\begin\{table\}[]
\\begin\{tabular\}\{lllll\}
            """)
        f.write( '\n'.join(lines)+'\\\\\n' )
        f.write("""
\\end\{tabular\}
\\end\{table\}
            """)
        f.close()

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
                key = f"{ds}_{level}"
                if(not key in best):
                    best[key] = {}

                for m in target_metrics:
                    dat = {}
                    for mode in evaluation_modes:
                        fname = f"{mode}_summary-report_test-model-{level}.tsv"
                        path = os.path.join(pathdir, fname)
                        df = pd.read_csv( path, sep= '\t')
                        aux = df[ (df['evaluation_metric'] == m) & (df['stats_agg_name'] == 'mean') ]
                        dat[mode] = aux.stats_agg_value.max()
                    odat = dict( sorted( dat.items(), key=lambda item: item[1], reverse=True ) )
                    best[key][m] = list(odat)[0]

        lines = []
        l = ["dataset", "entity", "level", "metric", "value"]
        lines.append(l)
        for k in best:
            ds, level = k.split('_')
            pathdir = os.path.join( indir.replace('__db__', ds), 'test', 'summary_reports' )
            for m in best[k]:
                mode = best[key][m]
                fname = f"{mode}_summary-report_test-model-{level}.tsv"
                path = os.path.join(pathdir, fname)
                df = pd.read_csv( path, sep= '\t')
                aux = df[ (df['evaluation_metric'] == m) & (df['stats_agg_name'] == 'mean') ]
                for i in aux.index:
                    entity = df.iloc[i, 0]
                    values = df.iloc[i, 4:]
                    for v in values:
                        l = [ ds, entity, level, level, m, v ]
                        l = '\t'.join( [ str(x) for x in l] )
                        lines.append(l)
        opath = os.path.join(self.out, "table_comp_sota.tsv")
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()

        m = 'f1-score'
        df = pd.read_csv( opath, sep= '\t')
        for ds in dss:
            aux = df[ (df["dataset"] == ds) && (df["metric"] == m) ]
            fig = px.box( aux, x="entity", y="value", color="level")
            opath = os.path.join(self.out, f'{ds}_{m}_comparison_sota.png')
            fig.write_image( opath )

    def run(self):
        self.wrap_picods_comp_metrics_reprod()
        self.wrap_bench_dss_eval_metrics_reprod()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/valout'
    i = ExplorationBenchmarkDss( odir )
    i.run()