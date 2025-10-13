import os
import re
import sys
import pickle

import rdflib

import pandas as pd
from tqdm import tqdm
import plotly.express as px
from scipy.stats import ranksums
from scipy import stats

from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD, BNode
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD, OWL

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class ExplorationSemanticResults:
    def __init__(self, fout):
        self.graph = rdflib.Graph()

        self.out = fout
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )

        '''
        Acquire computed data and put them together: 
        cp nerfairwf_experiments/trials/biobert-*-hypersearch-biobert-base-cased-v1.2-finetuned-ner/experiment_metadata/experiment_graph_biobert-*-hypersearch.ttl ./out_eda_semantic/data
        cp nerfairwf_experiments/trials/biobert-*-hypersearch-biobert-base-cased-v1.2-finetuned-ner/experiment_metadata/experiment_graph_biobert-*-hypersearch.ttl ./out_eda_semantic/data

        '''
        self._setup_namespaces()

    def _setup_namespaces(self):
        g = self.graph

        self.nerwf = Namespace("http://example.org/")
        g.bind("nerwf", self.nerwf)
        self.xmlpo = Namespace("https://w3id.org/ontouml-models/model/xhani2023xmlpo/") # https://github.com/OntoUML/ontouml-models/raw/refs/heads/master/models/xhani2023xmlpo/ontology.ttl
        g.bind("xmlpo", self.xmlpo)
        self.stato = Namespace("http://purl.obolibrary.org/obo/STATO_")
        g.bind("stato", self.stato)
        self.ncit = Namespace("http://purl.obolibrary.org/obo/NCIT_")
        g.bind("ncit", self.ncit)
        self.mesh = Namespace("http://id.nlm.nih.gov/mesh/")
        g.bind("mesh", self.mesh)
        self.slso = Namespace("http://purl.obolibrary.org/obo/SLSO_")
        g.bind("slso", self.slso)
        self.vcard = Namespace("http://www.w3.org/2006/vcard/ns#")
        g.bind("vcard", self.vcard)
        self.edam = Namespace("http://edamontology.org/")
        g.bind("edam", self.edam)
        self.nero = Namespace("http://www.cs.man.ac.uk/~stevensr/ontology/ner.owl#")
        g.bind("nero", self.nero)

        self.graph = g

    def load_graphs(self):
        g = self.graph

        indir = os.path.join(self.out, 'data')
        for f in os.listdir(indir):
            path = os.path.join(indir, f)
            g.parse(path)

        self.graph = g

    def count_new_classes_properties(self):
        g = self.graph

        meta = { 'classes': 'owl:Class', 'object_properties': 'OWL:ObjectProperty', 'data_properties': 'OWL:DatatypeProperty' }
        res = {}
        for k in meta:
            q = '''
    SELECT ( count( DISTINCT ?s ) as ?cnt)
    WHERE {
        ?s rdf:type %s .

    }
            ''' %( meta[k] )
            
            qres = g.query(q, initNs={'rdf': RDF, 'owl': OWL})
            for row in qres:
                print( f"{row.cnt} {k}" )
                res[k] = row.cnt
        
        return res

    def run(self):
        self.load_graphs()
        self.count_new_classes_properties()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/out_eda_semantic'
    i = ExplorationSemanticResults( odir )
    i.run()