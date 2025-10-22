import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from scipy.stats import friedmanchisquare

import argparse
import logging

class AnalysisStatisticalSignificance:
    
    def __init__(self):
        self._get_arguments()
        self._setup_out_folders()

        self.fready = os.path.join( self.logdir, "tasks_significance_analysis.ready")
        self.ready = os.path.exists( self.fready )
        if(self.ready):
            self.logger.info("----------- Prediction step skipped since it was already computed -----------")
            self.logger.info("----------- Prediction step ended -----------")
        
    def _get_arguments(self):
        parser = argparse.ArgumentParser(description='Significance Analysis')
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
        logf = os.path.join( self.logdir, "significance_analysis.log" )
        logging.basicConfig( filename=logf, encoding="utf-8", filemode="a", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
        self.logger = logging.getLogger('prediction')

        try:
            self.expid = self.config["identifier"]
            self.outDirRoot = self.config["outpath"]
            self.external_eval_data = None
            self.external_eval_agg = "median"
            if( 'external_eval_data' in self.config ):
                self.external_eval_data = self.config["external_eval_data"]
                if( "agg_function" in  self.external_eval_data ):
                    self.external_eval_agg = self.external_eval_data["agg_function"]

                '''
"external_eval_data": { 
    "mode": "per_entity",
    "evaluators": [ 
        { 
            "identifier": "ref6", 
            "level": "token", 
            "results": { "age": { "f1-score": [0.8, 0.8, 0.8, 0.8], "accuracy": [0.9, 0.9, 0.9, 0.9, 0.9] }, "outcome": { "f1-score": [0.8, 0.8, 0.8, 0.8], "accuracy": [0.9, 0.9, 0.9, 0.9, 0.9] } } 
        } 
    ] 
},
"external_eval_data": { 
    "mode": "global",
    "agg_function": "median",
    "evaluators": [ 
        { 
            "identifier": "ref6", 
            "level": "token", 
            "results": { "age": { "f1-score": 0.8, "accuracy": 0.9 }, "outcome": { "f1-score": 0.8, "accuracy": 0.9 } } 
        } 
    ] 
},
                '''

            self.logger.info("----------- Significance analysis step started -----------")
        except:
            raise Exception("Mandatory fields not found in config. file")

    def _setup_out_folders(self):
        self.out = os.path.join( self.outDirRoot, "significance_analysis" )
        if( not os.path.exists(self.out) ):
            os.makedirs( self.out )

    def _acquire_data(self):
        data = {}

        stages = ["training", "test"]
        for stage in stages:
            folders = glob.glob( os.path.join(self.outDirRoot, f"*-finetuned-*", stage, 'summary_reports') )

            evaluation_modes = ['seqeval-default', 'seqeval-strict', 'sk-with-prefix', 'sk-without-prefix']
            levels = ['token', 'word']
            for mode in evaluation_modes:
                for level in levels:
                    for folder in folders:
                        fid = os.path.dirname( folder + os.path.sep ).split( os.path.sep )[-3]

                        files = glob.glob( os.path.join(folder, f"{mode}_summary-report_*-l{level}.tsv") )
                        if( len(files) > 0 ):
                            for f in files:
                                df = pd.read_csv(f, sep='\t')
                                for i in df.index:
                                    evalMetric = df.loc[i, 'evaluation_metric']
                                    entity = df.loc[i, 'Entity']
                                    agg = df.loc[i, "stats_agg_name"]
                                    value = df.loc[i, "stats_agg_value"]
                                    values = df.iloc[i, 4:].values

                                    key = f"{stage}_{mode}_{level}"
                                    if( not key in data ):
                                        data[key] = {}
                                    if( not fid in data[key] ):
                                        data[key][fid] = {}

                                    data[key][fid][ f"{entity}#$@{evalMetric}" ] = values

                                    if( self.external_eval_agg == agg ):
                                        if( not evalMetric in data[key][fid] ):
                                            data[key][fid][evalMetric] = {}
                                        data[key][fid][evalMetric][entity] = value


        return data

    def _get_external_evaluation(self, level, dat):
        if( self.external_eval_data is not None ):
            mode = self.external_eval_data["mode"]
            for ext in self.external_eval_data["evaluators"]:
                if( ext["level"] == level ):
                    fid = ext["identifier"]
                    dat[fid] = {}
                    for entity in ext["results"]:
                        for evalMetric in ext["results"][entity]:
                            values = ext["results"][entity][evalMetric]

                            if( mode == 'per_entity'):
                                dat[fid][ f"{entity}#$@{evalMetric}" ] = values
                            else:
                                if( not evalMetric in dat[fid] ):
                                    dat[fid][evalMetric] = {}
                                dat[fid][evalMetric][entity] = values

        return dat

    def perform_stats_analysis_per_entity(self):
        self.logger.info("[Significance analysis step] Task (Performing statistical significance analysis) started -----------")
        
        data = self._acquire_data()

        for key in data:
            stage, mode, level = key.split('_')
            dat = data[key]
            
            opath = os.path.join( self.out, f'{key}_wilcoxon_analysis.tsv' )
            f = open( opath, 'w' )
            f.write("model_base\tmodel_comparison\tevaluation_metric\tentity\tp_value_wilcoxon\n")
            
            frivalues = {}
            gbfrivalues = {}
            ents = set()
            ref = None
            
            for i, model_base in enumerate(dat):
                # Setup for friedmanchisquare
                entity_metric = list(dat[model_base])[0]
                if( (entity_metric.find('#$@') != -1) ): # per entity mode
                    for entity_metric in dat[model_base]:
                        if( not entity_metric in frivalues ):
                            frivalues[entity_metric] = []
                        frivalues[entity_metric].append( dat[model_base][entity_metric] )
                else:
                    for evalMetric in dat[model_base]:
                        if( len(ents) == 0 ):
                            ents.update( list(dat[model_base][evalMetric]) )
                        if( not evalMetric in gbfrivalues ):
                            gbfrivalues[evalMetric] = []

                        elements = [ dat[model_base][evalMetric][e] for e in ents ]
                        if( ref is None ):
                            ref = elements
                        if( len(ref) == len(elements) ):
                            gbfrivalues[evalMetric].append( elements )

                # Setup for wilcoxon
                for j, model_compare in enumerate(dat):
                    if( i < j ):
                        entity_metric = list(dat[model_base])[0]
                        if( (entity_metric.find('#$@') != -1) ): # per entity mode
                            for entity_metric in dat[model_base]:
                                if( entity_metric in dat[model_compare] ):
                                    entity, metric = entity_metric.split('#$@')
                                    pvalue_wil = ranksums( dat[model_base][entity_metric], dat[model_compare][entity_metric] )
                                    f.write(f"{model_base}\t{model_compare}\t{metric}\t{entity}\t{ pvalue_wil.pvalue }\n")
                        else: # global mode
                            for evalMetric in dat[model_base]:
                                entsA = set(dat[model_base][evalMetric])
                                entsB = set(dat[model_compare][evalMetric])

                                valuesA = []
                                valuesB = []
                                inter = entsA.intersection(entsB)
                                for e in inter:
                                    valuesA.append( dat[model_base][evalMetric][e] )
                                    valuesB.append( dat[model_compare][evalMetric][e] )
                                
                                if(len(inter) > 1):
                                    entity = 'global'
                                    pvalue_wil = ranksums( valuesA, valuesB )
                                    fp.write(f"{model_base}\t{model_compare}\t{metric}\t{entity}\t{ pvalue_wil.pvalue }\n")

            f.close()
            fp.close()

            opath = os.path.join( self.out, f'{key}_friedman_analysis.tsv' )
            f = open( opath, 'w' )
            f.write("evaluation_metric\tentity\tp_value_friedman\n")
            body = []
            for k in frivalues:
                if( len( frivalues[k] ) >= 3 ):
                    entity, metric = k.split('#$@')
                    res = friedmanchisquare( *( for x in frivalues[k] ) )
                    body.append( '\t'.join( [ metric, entity, str(res.pvalue) ] ) )
            f.write( '\n'.join(body)+'\n' )
            
            entity = 'global'
            body = []
            for evalMetric in gbfrivalues:
                if( len( gbfrivalues[evalMetric] ) >= 3 ):
                    res = friedmanchisquare( *( for x in gbfrivalues[evalMetric] ) )
                    body.append( '\t'.join( [ evalMetric, entity, str(res.pvalue) ] ) )
            f.write( '\n'.join(body)+'\n' )
            f.close()

        self.logger.info("[Significance analysis step] Task (Performing statistical significance analysis) ended -----------")
       
    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Significance analysis step ended -----------")
    
    def run(self):
        self.perform_stats_analysis_per_entity()
 
if( __name__ == "__main__" ):
    i = AnalysisStatisticalSignificance( )
    if( not i.ready ):
        i.run()