import os
import sys
import json
from uuid import uuid4
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD, OWL

class SemanticDescription:
	
	def __init__(self, fout):
		config = json.load( open('/aloy/home/ymartins/match_clinical_trial/experiments/config_biobert.json','r') )
		if( 'experiment_metadata' in config ):
		self.exp_metadata = config['experiment_metadata']

	
	def gen_id(self):
		_id = uuid4().hex
		return _id

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

		self.graph = g
		
	def _define_new_onto_elements(self):
		g = self.graph
		
		# Defining specific types of Experiment
		g.add( ( self.nerwf.MLExperiment, RDFS.subClassOf, self.xmlpo.Experiment ) )
		g.add(( self.nerwf.MLExperiment, RDFS.label,   Literal("MLExperiment", lang="en")))
		g.add( ( self.nerwf.NLPExperiment, RDFS.subClassOf, self.nerwf.MLExperiment ) )
		g.add(( self.nerwf.NLPExperiment, RDFS.label,   Literal("NLPExperiment", lang="en")))

		# Defining Predictive Learning Models LLM model as a sub class of the general Model class
		g.add( ( self.mesh.D000098412, RDFS.subClassOf, self.ncit.C43383 ) ) 
		# Defining LLM model as a sub class of the Predictive Learning Models class
		g.add( ( self.mesh.D000098342, RDFS.subClassOf, self.mesh.D000098342 ) ) 
		
		g.add(( self.nerwf.hasDataset, RDF.type, OWL.ObjectProperty) )
		g.add(( self.nerwf.hasDataset, RDFS.domain, self.xmlpo.Experiment ))
		g.add(( self.nerwf.hasDataset, RDFS.range,  self.xmlpo.Dataset))
		g.add(( self.nerwf.hasDataset, RDFS.label,   Literal("hasDataset", lang="en")))
		g.add(( self.nerwf.hasDataset, RDFS.comment,   Literal("Datasets associated to an experiment", lang="en")))

		g.add(( self.nerwf.executedBy, RDF.type, OWL.ObjectProperty) )
		g.add(( self.nerwf.executedBy, RDFS.label,   Literal("executedBy", lang="en")))
		g.add(( self.nerwf.executedBy, RDFS.comment,   Literal("Experiments executed by workflow, a step of workflow can also be executed by an algorithm", lang="en")))

		g.add(( self.nerwf.describedBy, RDF.type, OWL.ObjectProperty) )
		g.add(( self.nerwf.describedBy, RDFS.domain, self.OWL.Class ))
		g.add(( self.nerwf.describedBy, RDFS.range,  self.xmlpo.Quality))
		g.add(( self.nerwf.describedBy, RDFS.label,   Literal("describedBy", lang="en")))
		g.add(( self.nerwf.describedBy, RDFS.comment,   Literal("Classes may be described by an object of Quality class of its subclasses that specify characteristics", lang="en")))
		
		g.add(( self.nerwf.finetunedFrom, RDF.type, OWL.ObjectProperty) )
		g.add(( self.nerwf.finetunedFrom, RDFS.domain, self.mesh.D000098342 )) # LLM
		g.add(( self.nerwf.finetunedFrom, RDFS.range,  self.mesh.D000098342))
		g.add(( self.nerwf.finetunedFrom, RDFS.label,   Literal("finetunedFrom", lang="en")))
		g.add(( self.nerwf.finetunedFrom, RDFS.comment,   Literal("Specify whether a new produced ML model was refined from an LLM model", lang="en")))
		
		g.add(( self.nerwf.hasParameter, RDF.type, OWL.ObjectProperty) )
		g.add(( self.nerwf.hasParameter, RDFS.domain, self.ncit.C43383 )) # Model
		g.add(( self.nerwf.hasParameter, RDFS.range,  self.xmlpo.parameterSettings))
		g.add(( self.nerwf.hasParameter, RDFS.label,   Literal("hasParameter", lang="en")))
		g.add(( self.nerwf.hasParameter, RDFS.comment,   Literal("It can be used to specify the model parameters (hyper parameters)", lang="en")))
		
		g.add(( self.nerwf.hasReplicateNumber, RDF.type, OWL.DatatypeProperty) )
		g.add(( self.nerwf.hasReplicateNumber, RDFS.domain, self.OWL.Class )) 
		g.add(( self.nerwf.hasReplicateNumber, RDFS.range,  XSD.integer))
		g.add(( self.nerwf.hasReplicateNumber, RDFS.label,   Literal("hasParameter", lang="en")))
		g.add(( self.nerwf.hasReplicateNumber, RDFS.comment,   Literal("It can be used to specify the model parameters (hyper parameters)", lang="en")))
		
		self.graph = g

	def _describe_experiment(self):
		g = self.graph
		g.add( ( self.nerwf['exp_'+uuid4().hex], RDF.type, self.nerwf.NLPExperiment ) )

		self.graph = g

	def _describe_data_preprocessing(self):
		g = self.graph
		expid = str(uuid4())
		g.add( ( self.nerwf[f"experiment/{expid}"], RDF.type, self.xmlpo.Experiment) )
		
		self.graph = g

	def run(self):
		'''
		experiment
			model1 -> hyperparams

		workflow
			containsProcedure
				Preprocess (operation)
					executes PreprocessImplementation

				Train
					executes SupervisedLearningApproachImplementation
					SupervisedLearningApproachImplementation implements classificationAlgorithm (mlapproach)
					NEREvaluationMeasure subclassof ClassPredictionEvaluationMeasure
					classificationAlgorithm hasOutput DatasetOutTrain
				
				MLModelEvaluation executes ModelEvaluationTechnique
					underContext 'Train'
					takeDataInput instanceDataset (DatasetOutTrain)
					generatesScoreDataset score_dataset
					self:hasScore self:NEREvaluationMeasure (nscore1)
				
				nscore1 self:belongsToEntity "intervention"
				nscore1 fromEvaluationMetric f1, acc, mcc
				nscore1 isAggregated true
				nscore1 aggregatedBy max (C), min, mean (ncit:C53319), std (ncit:C53322), median (ncit:c28007)
				nscore1 hasValue 0.8

				ModelEvaluationTechnique describedBy ModelEvaluationCharacteristic
				modelEvaluation1 a ModelEvaluationCharacteristic
				modelEvaluation1 MLEvaluationTechniqueName sklearn_without-prefix
				modelEvaluation1 MLEvaluationTechniqueDescription sk learn without the prefixes

				Test
				Prediction
					generatesDataset

		preprocess
			instance dataset (LabeledDataset)
				rdfs
		train

			self:generatesDataset score dataset (Dataset class)
			self:generatesModel model 1
			self:generatedBy trainSlaImpl1 (step implementation class)
			self:hasScore self:NEREvaluationMeasure
				self:belongsTo self:NEREntity
				fromEvaluationMetric f1, acc, mcc
				isAggregated true
				aggregatedBy max (C), min, mean (ncit:C53319), std (ncit:C53322), median (ncit:c28007)
				hasValue 0.8
		test
		'''