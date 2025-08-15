import os
import sys
import json
from uuid import uuid4
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD

class SemanticDescription:
	
	def __init__(self, fout):


	def _setup_namespaces(self):
		self.nerwf = Namespace("http://example.org/")
		g.bind("nerwf", self.nerwf)
		self.xmlpo = Namespace("http://ExplainableMachineLearningPipelineOntology#")
		g.bind("xmlpo", self.xmlpo)

	def _describe_items(self):
		expid = str(uuid4())
		g.add( ( self.nerwf[f"experiment/{expid}"], RDF.type, SCHEMA.Book) )