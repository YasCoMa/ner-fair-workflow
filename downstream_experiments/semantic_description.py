import os
import sys
import json
from uuid import uuid4
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD

class SemanticDescription:
	
	def __init__(self, fout):
		config = json.load( open('/aloy/home/ymartins/match_clinical_trial/experiments/config_biobert.json','r') )

	def _setup_namespaces(self):
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
		
	def _define_new_onto_elements(self):


	def _describe_items(self):
		expid = str(uuid4())
		g.add( ( self.nerwf[f"experiment/{expid}"], RDF.type, self.xmlpo.Experiment) )