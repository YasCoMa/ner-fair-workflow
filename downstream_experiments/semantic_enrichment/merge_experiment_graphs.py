import os
import glob
import json
import shutil
import rdflib

from tqdm import tqdm

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class IntegrateExperimentKnowledgeGraphs:
    def __init__(self, exp_folder, fout):
        self.graph = rdflib.Graph()

        self.ds_folder = exp_folder
        self.out =  fout
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )

        self.dss =  [ "bc5cdr", "ncbi", "biored", "chia_without_scope", "chia_without_scope", "merged_train" ]

    def rerun_meta_enrichment(self):
        log_folder= os.path.join(self.exp_folder, "trials", "logs")
        os.system( "rm %s/*semantic_description.ready" %(log_folder) )

        dss = self.dss
        for ds in dss:
            cmds.append( "nextflow run %s/main.nf --dataDir %s/trials/ --runningConfig %s/configs/config_%s.json --mode 'metadata_enrichment'" %(root_path, self.exp_folder, self.exp_folder, ds) )

        i = 1
        for cmd in cmds:
            print(i, '/', len(cmds))
            os.system(cmd)
            i += 1

    def _copy_rdf_files(self):
        dss = self.dss
        configs = []
        for ds in dss:
            configs.append( "%s/configs/config_%s.json" %(root_path, ds) )

        for c in configs:
            config = json.load( open(c, 'r') )
            task = 'ner'
            model_checkpoint = config["pretrained_model"]
            expid = config["identifier"]

            model_name = model_checkpoint.split("/")[-1]
            fout = config["outpath"]
            outDir = os.path.join(fout, f"{expid}-{model_name}-finetuned-{task}" )
            indir = os.path.join( outDir, "experiment_metadata" )
            outdir = os.path.join(self.out, 'data')
            os.system( "cp %s/* %s/" %(indir, outdir) )

    def load_graphs(self):
        self._copy_rdf_files()

        g = self.graph

        indir = os.path.join(self.out, 'data')
        for f in os.listdir(indir):
            if( f.endswith('ttl') ):
                path = os.path.join(indir, f)
                g.parse(path)

        opath = os.path.join( self.out, 'all_nerfair_graph.ttl')
        g.serialize( destination=opath )
        opath = os.path.join( self.out, 'all_nerfair_graph.xml')
        g.serialize( destination=opath, format="xml" )

        txt = open( opath ).read()
        txt = txt.replace('<rdf:RDF', '<rdf:RDF xmlns="https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#"\nxml:base="https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#"\n')
        f = open( opath,'w')
        f.write(txt)
        f.close()

    def run(self):
        self.rerun_meta_enrichment()
        self.load_graphs()

if( __name__ == "__main__" ):
    exp_folder = "./preprocessed_datasets_for_workflow"
    fout = "./merged_experiment_graphs"
    if( len(sys.argv) > 1 ):
        fout = sys.argv[1]
    i = IntegrateExperimentKnowledgeGraphs( exp_folder, fout )
    i.run()