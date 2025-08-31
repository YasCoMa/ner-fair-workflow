import os
import glob
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
                                    values = df.iloc[i, 4:].values

                                    key = f"{stage}_{mode}_{level}"
                                    if( not key in data ):
                                        data[key] = {}
                                    if( not fid in data[key] ):
                                        data[key][fid] = {}

                                    data[key][fid][ f"{entity}#$@{evalMetric}" ] = values

        return data

    def perform_stats_analysis(self):
        self.logger.info("[Significance analysis step] Task (Performing statistical significance analysis) started -----------")
        
        data = self._acquire_data()

        for key in data:
            opath = os.path.join( self.out, f'{key}_analysis.tsv' )
            dat = data[key]
            f = open( opath, 'w' )
            f.write("model_base\tmodel_comparison\tevaluation_metric\tentity\tp_value_wilcoxon\tp_value_friedman\n")
            for i, model_base in enumerate(dat):
                for j, model_compare in enumerate(dat):
                    if( i < j ):
                        for entity_metric in dat[model_base]:
                            entity, metric = entity_metric.split('#$@')
                            pvalue_wil = ranksums( dat[model_base][entity_metric], dat[model_compare][entity_metric] )
                            pvalue_fried = friedmanchisquare( dat[model_base][entity_metric], dat[model_compare][entity_metric] )
                            f.write(f"{model_base}\t{model_compare}\t{metric}\t{entity}\t{ pvalue_wil.pvalue }\t{ pvalue_fried.pvalue }\n")

            f.close()

        self.logger.info("[Significance analysis step] Task (Performing statistical significance analysis) ended -----------")
       
    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Significance analysis step ended -----------")
    
    def run(self):
        self.perform_stats_analysis()
 
if( __name__ == "__main__" ):
    i = AnalysisStatisticalSignificance( )
    if( not i.ready ):
        i.run()