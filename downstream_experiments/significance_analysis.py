import os
import glob
import pandas as pd
from scipy.stats import ranksums
from scipy.stats import friedmanchisquare

class AnalysisStatisticalSignificance:
	def __init__(self):
		self.stage = 'test'
		self.root_path = '/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/'
		self.out = os.path.join( self.root_path, 'statistical_comparison' )
		if( not os.path.isdir( self.out ) ):
			os.makedirs( self.out )

	def _acquire_data(self):
		data = {}

		folders = glob.glob( os.path.join(self.root_path, f"*-finetuned-*", self.stage, 'summary_reports') )

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

	                            key = f"{mode}_{level}"
	                            if( not key in data ):
	                            	data[key] = {}
	                            if( not fid in data[key] ):
	                            	data[key][fid] = {}

	                            data[key][fid][ f"{entity}#$@{evalMetric}" ] = values

		return data

	def perform_stats_analysis(self):
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

	def run(self):
		self.perform_stats_analysis()

if( __name__ == "__main__" ):
    i = AnalysisStatisticalSignificance(  )
    i.run()
