import os
import sys
import json

import rdflib
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD, BNode
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD, OWL
from rdflib.collection import Collection


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.graphs import RdfGraph
from langchain_community.chains.graph_qa.sparql import GraphSparqlQAChain

import requests

from llm_utils import create_nlq_vdb
from franz.openrdf.connect import ag_connect
from franz.openrdf.sail.allegrographserver import AllegroGraphServer
from franz.openrdf.connect import ag_connect
from franz.openrdf.rio.rdfformat import RDFFormat
from franz.openrdf.repository.repository import Repository

REPO='nerwfexp'
AGRAPH_HOST = os.environ.get('AGRAPH_HOST')
AGRAPH_PORT = int(os.environ.get('AGRAPH_PORT', '10035'))
AGRAPH_USER = os.environ.get('AGRAPH_USER')
AGRAPH_PASSWORD = os.environ.get('AGRAPH_PASSWORD')

class EdaSemanticLLM:
    def __init__(self, fout):
        self.graph = rdflib.Graph()

        self.out = fout
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )



    def rerun_meta_enrichment(self):
        logs = ["/aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/", "/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/trials/"]
        for l in logs:
            os.system( "rm %s/logs/*semantic_description.ready" %(l) )

        dss = [ "bc5cdr", "ncbi", "biored", "chiads"]
        cmds = [
        "nextflow run /aloy/home/ymartins/match_clinical_trial/ner_subproj/main.nf --dataDir /aloy/home/ymartins/match_clinical_trial/experiments/biobert_trial/ --runningConfig /aloy/home/ymartins/match_clinical_trial/experiments/config_biobert_hypersearch.json --mode 'metadata_enrichment'",
        "nextflow run /aloy/home/ymartins/match_clinical_trial/ner_subproj/main.nf --dataDir /aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/trials/ --runningConfig /aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/configs/config_merged_train.json --mode 'metadata_enrichment'"
        ]
        for ds in dss:
            cmds.append( "nextflow run /aloy/home/ymartins/match_clinical_trial/ner_subproj/main.nf --dataDir /aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/trials/ --runningConfig /aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/configs/config_%s.json --mode 'metadata_enrichment'" %(ds) )

        i = 1
        for cmd in cmds:
            print(i, '/', len(cmds))
            os.system(cmd)
            i += 1

    def _copy_rdf_files(self):
        dss = [ "bc5cdr", "ncbi", "biored", "chiads"]
        configs = ["/aloy/home/ymartins/match_clinical_trial/experiments/config_biobert_hypersearch.json", "/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/configs/config_merged_train.json"]
        for ds in dss:
            configs.append( "/aloy/home/ymartins/match_clinical_trial/nerfairwf_experiments/configs/config_%s.json" %(ds) )

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
        txt = txt.replace('<rdf:RDF','<rdf:RDF xmlns="https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#"\nxml:base="https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#"\n')
        f = open( opath,'w')
        f.write(txt)
        f.close()

        self.graph = g

    def export_to_alegrograph(self):
        # /home/ymartins/agraph/bin/agraph-control --config /home/ymartins/agraph/lib/agraph.cfg start

        print("Initializing repository")
        server = AllegroGraphServer(AGRAPH_HOST, AGRAPH_PORT, AGRAPH_USER, AGRAPH_PASSWORD)
        catalog = server.openCatalog('')

        mode = Repository.RENEW
        _repo = catalog.getRepository(REPO, mode)
        _repo.initialize()
        conn = _repo.getConnection()

        print("Loading data")
        opath = os.path.join( self.out, 'all_nerfair_graph.ttl')
        conn.addFile( opath, None, format=RDFFormat.TURTLE, context=None)

    def test_ag_llm(self):
        NLQ_VDB='nerwfexp_vdb'
        EMBEDDING_MODEL="mxbai-embed-large"
        EMBEDDER="ollama"
        OPENAI_API_KEY=''

        conn = ag_connect(
            REPO,
            clear=True,
            user=AGRAPH_USER,
            password=AGRAPH_PASSWORD,
            host=AGRAPH_HOST,
            port=AGRAPH_PORT
        )
        opath = os.path.join( self.out, 'all_nerfair_graph.ttl')
        conn.addFile( opath, None, format=RDFFormat.TURTLE, context=None)

        #connecting to nlq vdb
        nlq_conn = create_nlq_vdb(
            REPO,
            conn,
            NLQ_VDB,
            host=AGRAPH_HOST,
            port=AGRAPH_PORT,
            user=AGRAPH_USER,
            password=AGRAPH_PASSWORD,
            openai_api_key=OPENAI_API_KEY,
            embedder=EMBEDDER,
            embedding_model=EMBEDDING_MODEL
        )

        prompt = "show me 10 triples"

        result = conn.execute_nl_query(
            prompt,
            NLQ_VDB
        )
        print(result)

    def test_llama_agexec(self):
scheme = 'http'
ollama_host = 'localhost'
ollama_port = '11434'
model = 'llama3.2:latest'

PREFIXES = f"""
PREFIX franzOption_llmVendor: <franz:ollama>  
PREFIX franzOption_llmScheme: <franz:{scheme}>  
PREFIX franzOption_llmHost: <franz:{ollama_host}>  
PREFIX franzOption_llmPort: <franz:{ollama_port}>  
PREFIX franzOption_llmChatModel: <franz:{model}>  
 
PREFIX llm: <http://franz.com/ns/allegrograph/8.0.0/llm/>   
"""

conn = ag_connect(
    REPO,
    clear=True,
    user=AGRAPH_USER,
    password=AGRAPH_PASSWORD,
    host=AGRAPH_HOST,
    port=AGRAPH_PORT
)
opath = os.path.join( self.out, 'all_nerfair_graph.ttl')
conn.addFile( opath, None, format=RDFFormat.TURTLE, context=None)

query_string = f"""
    {PREFIXES}

    SELECT ?entity WHERE {{
        ?entity llm:response "Get the distinct names of named entities used in each experiment" }}"""
res = conn.executeTupleQuery(query_string)
print(res)

    def run(self):
        #self.rerun_meta_enrichment()
        #self.load_graphs()

        #self.export_to_alegrograph()
        #self.test_ag_llm()
        self.test_llama_agexec()

if( __name__ == "__main__" ):
    odir = '../paper_files/out_eda_semantic'
    odir = '/aloy/home/ymartins/match_clinical_trial/out_eda_semantic'
    i = EdaSemanticLLM( odir )
    i.run()
