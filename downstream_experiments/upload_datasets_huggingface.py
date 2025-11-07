import os
import json
from huggingface_hub import login
from datasets import Dataset, DatasetDict, load_from_disk

class UploadDatasetsHUb:
    def __init__(self):
        self.hfuser = 'yasmmin'
        login()
    
    def upload(self):
        dss = [ "bc5cdr", "ncbi", "biored", "chiads"]

        configs = ["/aloy/home/ymartins/match_clinical_trial/experiments/config_biobert_hypersearch.json", "/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/configs/config_merged_train.json", "/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/configs/config_merged_test.json"]

        for ds in dss:
            configs.append( "/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/configs/config_%s.json" %(ds) )

        for c in configs:
            aux = c.split('/')[-1].replace('config_','').replace('.json','')
            if( aux.startswith('biobert') ):
                aux = 'pico-human-corpus'
            reponame = aux + "_nerfair_processed"

            config = json.load( open(c, 'r') )
            task = 'ner'
            model_checkpoint = config["pretrained_model"]
            expid = config["identifier"]

            model_name = model_checkpoint.split("/")[-1]
            fout = config["outpath"]
            outDir = os.path.join(fout, f"{expid}-{model_name}-finetuned-{task}" )
            indir = os.path.join( outDir, "preprocessing" )

            for d in os.listdir(indir):
                path = os.path.join(indir, d)
                if( os.path.isdir( path ) and d.startswith('dataset_') ):
                    datasets = load_from_disk( path )
                    print(reponame)
                    datasets.push_to_hub( f"{self.hfuser}/{reponame}" )

    def run(self):
        self.upload()

if( __name__ == "__main__" ):
    i = UploadDatasetsHUb( )
    i.run()