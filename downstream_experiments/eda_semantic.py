import os
import re
import sys
import json
import time
import pickle

import rdflib

import pandas as pd
from tqdm import tqdm
import plotly.express as px
from scipy.stats import ranksums
from scipy import stats

#from owlready2 import *
import owlready2

from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD, BNode
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD, OWL
from rdflib.collection import Collection

os.environ["GOOGLE_API_KEY"] = "AIzaSyDV66WrLSzULt9PHN2gvqnx0qPmZsBHniI"
os.environ["OPENAI_API_KEY"] = "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"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.graphs import RdfGraph
from langchain_community.chains.graph_qa.sparql import GraphSparqlQAChain

'''
dependencies:
langchain==0.3.27
langchain-ollama==0.3.10
'''

#from langchain_openai import ChatOpenAI
#from langchain_core.messages import HumanMessage

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class ExplorationSemanticResults:
    def __init__(self, fout):
        self.google_model = "gemini-2.5-flash"
        self.llama_model ='llama3.2'

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

    def gen_id(self, prefix):
        _id = uuid4().hex
        return f'{prefix}_{_id}'

    def _setup_namespaces(self):
        g = self.graph

        self.nerwf = Namespace("https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfairwf_ontology.owl#")
        g.bind("", self.nerwf)
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
        
    def __define_tagging_format_instances(self):
        g = self.graph

        g.add( ( self.nerwf.nlpformat_io, RDF.type, self.nerwf.TaggingFormat ) )
        g.add( ( self.nerwf.nlpformat_io, RDFS.label, Literal( "IO" ) ) )
        
        g.add( ( self.nerwf.nlpformat_iob, RDF.type, self.nerwf.TaggingFormat ) )
        g.add( ( self.nerwf.nlpformat_iob, RDFS.label, Literal( "IOB" ) ) )
        
        g.add( ( self.nerwf.nlpformat_bioes, RDF.type, self.nerwf.TaggingFormat ) )
        g.add( ( self.nerwf.nlpformat_bioes, RDFS.label, Literal( "BIOES" ) ) )

        self.graph = g
        
    def _define_new_onto_elements(self):
        g = self.graph
        
        # Declaring ontology and license
        onto = URIRef("http://NERMachineLearningPipelineOntology")
        g.add( ( onto, RDF.type, OWL.Ontology ) )
        g.add( ( onto, DCTERMS.license, URIRef("http://creativecommons.org/licenses/by/4.0/") ) )
        g.add(( onto, RDFS.comment, Literal("NER ML ontology is licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).", lang="en") ) )
        
        # Defining specific types of Experiment
        g.add(( self.xmlpo.ClassPredictionEvaluationMeasure, RDF.type, OWL.Class) )
        g.add(( self.xmlpo.ClassPredictionEvaluationMeasure, RDFS.label,   Literal("ClassPredictionEvaluationMeasure", lang="en")))
        g.add(( self.xmlpo.ClassPredictionEvaluationMeasure, RDFS.comment,   Literal("Measure to evaluate the prediction", lang="en")))
        
        g.add(( self.edam.format_3862, RDF.type, OWL.Class) )
        g.add(( self.edam.format_3862, RDFS.label,   Literal("NLP annotation format", lang="en")))
        g.add(( self.edam.format_3862, RDFS.comment,   Literal("Defines the specific format for NLP", lang="en")))
        
        g.add(( self.edam.operation_0004, RDF.type, OWL.Class) )
        g.add(( self.edam.operation_0004, RDFS.label,   Literal("Operation", lang="en")))
        g.add(( self.edam.operation_0004, RDFS.comment,   Literal("Defines the certain execution procedure, it can be a function, a computational method, etc.", lang="en")))
        
        g.add(( self.xmlpo.Operation, RDF.type, OWL.Class) )
        g.add(( self.xmlpo.Operation, RDFS.label,   Literal("Operation", lang="en")))
        g.add(( self.xmlpo.Operation, RDFS.comment,   Literal("Defines a procedure inserted as part of a pipeline", lang="en")))
        
        g.add(( self.xmlpo.Result, RDF.type, OWL.Class) )
        g.add(( self.xmlpo.Result, RDFS.label,   Literal("Result", lang="en")))
        g.add(( self.xmlpo.Result, RDFS.comment,   Literal("Refers to the experiment result", lang="en")))
        
        g.add(( self.xmlpo.Quality, RDF.type, OWL.Class) )
        g.add(( self.xmlpo.Quality, RDFS.label,   Literal("Quality", lang="en")))
        g.add(( self.xmlpo.Quality, RDFS.comment,   Literal("Refers to the parent class of the descriptive subclasses that characterizes dataset, model evaluation, parameters of operations, etc.", lang="en")))
        
        g.add(( self.xmlpo.PreprocessedData, RDF.type, OWL.Class) )
        g.add(( self.xmlpo.PreprocessedData, RDFS.label,   Literal("PreprocessedData", lang="en")))
        g.add(( self.xmlpo.PreprocessedData, RDFS.comment,   Literal("Refers to the data ready to enter in the model evaluation", lang="en")))
        
        g.add(( self.nerwf.MLExperiment, RDF.type, OWL.Class) )
        g.add( ( self.nerwf.MLExperiment, RDFS.subClassOf, self.xmlpo.Experiment ) )
        g.add(( self.nerwf.MLExperiment, RDFS.label,   Literal("MLExperiment", lang="en")))
        g.add(( self.nerwf.MLExperiment, RDFS.comment,   Literal("Defines the specification of computational experiment related to machine learning", lang="en")))
        
        g.add(( self.nerwf.NLPExperiment, RDF.type, OWL.Class) )
        g.add( ( self.nerwf.NLPExperiment, RDFS.subClassOf, self.nerwf.MLExperiment ) )
        g.add(( self.nerwf.NLPExperiment, RDFS.label,   Literal("NLPExperiment", lang="en")))
        g.add(( self.nerwf.NLPExperiment, RDFS.comment,   Literal("Defines the specific computational experiment related to NLP", lang="en")))
        
        g.add(( self.nerwf.NEREvaluationMeasure, RDF.type, OWL.Class) )
        g.add( ( self.nerwf.NEREvaluationMeasure, RDFS.subClassOf, self.xmlpo.ClassPredictionEvaluationMeasure ) )
        g.add(( self.nerwf.NEREvaluationMeasure, RDFS.label,   Literal("NEREvaluationMeasure", lang="en")))
        g.add(( self.nerwf.NEREvaluationMeasure, RDFS.comment,   Literal("Mode of evaluation specific for NER tasks, like seqeval or scikit learn", lang="en")))
        
        g.add(( self.nerwf.ValidationSet, RDF.type, OWL.Class) )
        g.add( ( self.nerwf.ValidationSet, RDFS.subClassOf, self.xmlpo.PreprocessedData ) )
        g.add(( self.nerwf.ValidationSet, RDFS.label,   Literal("ValidationSet", lang="en")))
        g.add(( self.nerwf.ValidationSet, RDFS.comment,   Literal("Defines the third type of data split used during the training for the validation batch examples", lang="en")))
        
        g.add(( self.nerwf.ValidationSet, OWL.disjointWith, self.xmlpo.TrainSet ) )
        g.add(( self.nerwf.ValidationSet, OWL.disjointWith, self.xmlpo.TestSet ) )
        
        g.add(( self.nerwf.TaggingFormat, RDF.type, OWL.Class) )
        g.add( ( self.nerwf.TaggingFormat, RDFS.subClassOf, self.edam.format_3862 ) ) # NLP annotation format class from edam (format branch)
        g.add(( self.nerwf.TaggingFormat, RDFS.label,   Literal("TaggingFormat", lang="en")))
        g.add(( self.nerwf.TaggingFormat, RDFS.comment,   Literal("Defines the style of annotating the prefixes of the NER entties assigned to the tokens in the text", lang="en")))
        self.__define_tagging_format_instances()

        g.add(( self.nerwf.SummaryPrediction, RDF.type, OWL.Class) )
        g.add( ( self.nerwf.SummaryPrediction, RDFS.subClassOf, self.xmlpo.Result ) )
        g.add(( self.nerwf.SummaryPrediction, RDFS.label,   Literal("SummaryPrediction", lang="en")))
        g.add(( self.nerwf.SummaryPrediction, RDFS.comment,   Literal("Defines a specific type of result, in this case the one that summarizes the model replicate consensus", lang="en")))
        
        # Defining MLModel as a sub class of the general Model class
        g.add( ( self.xmlpo.MLModel, RDFS.subClassOf, self.ncit.C43383 ) ) 
        # Defining Predictive Learning Models model as a sub class of the MLModel class
        g.add( ( self.mesh.D000098412, RDFS.subClassOf, self.xmlpo.MLModel ) ) 
        # Defining LLM model as a sub class of the Predictive Learning Models class
        g.add( ( self.mesh.D000098342, RDFS.subClassOf, self.mesh.D000098412 ) ) 
        # Defining that Operation in EDAM has the same Meaning as in xmlpo
        g.add(( self.xmlpo.Operation, OWL.sameAs,  self.edam.operation_0004 )) # Operation class
        
        # --------- Object Properties
        g.add(( self.nerwf.containsTargetEntity, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.containsTargetEntity, RDFS.domain, self.nerwf.NLPExperiment ))
        g.add(( self.nerwf.containsTargetEntity, RDFS.range,  self.nero.NamedEntity ))
        g.add(( self.nerwf.containsTargetEntity, RDFS.label,   Literal("containsTargetEntity", lang="en")))
        g.add(( self.nerwf.containsTargetEntity, RDFS.comment,   Literal("Defines the link between the NER task experiment and its target entities that the models will be trained to predict", lang="en")))

        g.add(( self.nerwf.executedBy, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.executedBy, RDFS.domain, self.nerwf.NLPExperiment ))
        g.add(( self.nerwf.executedBy, RDFS.range,  self.xmlpo.workflow ))
        g.add(( self.nerwf.executedBy, RDFS.label,   Literal("executedBy", lang="en")))
        g.add(( self.nerwf.executedBy, RDFS.comment,   Literal("Experiments executed by workflow, a step of workflow can also be executed by an algorithm", lang="en")))

        g.add(( self.nerwf.executesExperiment, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.executesExperiment, OWL.inverseOf, self.nerwf.executedBy ) )
        g.add(( self.nerwf.executesExperiment, RDFS.domain,  self.xmlpo.workflow ))
        g.add(( self.nerwf.executesExperiment, RDFS.range, self.nerwf.NLPExperiment ))
        g.add(( self.nerwf.executesExperiment, RDFS.label,   Literal("executesExperiment", lang="en")))
        
        g.add(( self.nerwf.describedBy, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.describedBy, RDFS.domain, OWL.Class ))
        g.add(( self.nerwf.describedBy, RDFS.range,  self.xmlpo.Quality))
        g.add(( self.nerwf.describedBy, RDFS.label,   Literal("describedBy", lang="en")))
        g.add(( self.nerwf.describedBy, RDFS.comment,   Literal("Classes may be described by an object of Quality class of its subclasses that specify characteristics", lang="en")))
        
        g.add(( self.nerwf.describesFeatureOf, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.describesFeatureOf, OWL.inverseOf, self.nerwf.describedBy ) )
        g.add(( self.nerwf.describesFeatureOf, RDFS.domain,  self.xmlpo.Quality ))
        g.add(( self.nerwf.describesFeatureOf, RDFS.range, OWL.Class ))
        g.add(( self.nerwf.describesFeatureOf, RDFS.label,   Literal("describesFeatureOf", lang="en")))
        
        g.add(( self.nerwf.finetunedFrom, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.finetunedFrom, RDF.type, OWL.TransitiveProperty) )
        g.add(( self.nerwf.finetunedFrom, RDFS.domain, self.mesh.D000098342 )) # LLM
        g.add(( self.nerwf.finetunedFrom, RDFS.range,  self.mesh.D000098342))
        g.add(( self.nerwf.finetunedFrom, RDFS.label,   Literal("finetunedFrom", lang="en")))
        g.add(( self.nerwf.finetunedFrom, RDFS.comment,   Literal("Specify whether a new produced ML model was refined from an LLM model", lang="en")))
        
        g.add(( self.nerwf.hasParameter, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.hasParameter, RDFS.domain, self.ncit.C43383 )) # Model
        g.add(( self.nerwf.hasParameter, RDFS.range,  self.xmlpo.parameterSettings))
        g.add(( self.nerwf.hasParameter, RDFS.label,   Literal("hasParameter", lang="en")))
        g.add(( self.nerwf.hasParameter, RDFS.comment,   Literal("It can be used to specify the model parameters (hyper parameters)", lang="en")))
        
        g.add(( self.nerwf.isParameterOf, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.isParameterOf, OWL.inverseOf, self.nerwf.hasParameter ) )
        g.add(( self.nerwf.isParameterOf, RDFS.domain,  self.xmlpo.parameterSettings ))
        g.add(( self.nerwf.isParameterOf, RDFS.range, self.ncit.C43383 ))
        g.add(( self.nerwf.isParameterOf, RDFS.label,   Literal("isParameterOf", lang="en")))
        
        g.add(( self.nerwf.fromEvaluationMetric, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.fromEvaluationMetric, RDFS.domain, self.nerwf.NEREvaluationMeasure )) # Model
        g.add(( self.nerwf.fromEvaluationMetric, RDFS.range,  self.stato["0000039"])) # stato statistic
        g.add(( self.nerwf.fromEvaluationMetric, RDFS.label,   Literal("fromEvaluationMetric", lang="en")))
        g.add(( self.nerwf.fromEvaluationMetric, RDFS.comment,   Literal("It can be used to specify the model parameters (hyper parameters)", lang="en")))
        
        g.add(( self.nerwf.containsProcedure, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.containsProcedure, RDFS.domain, self.xmlpo.workflow )) # Model
        g.add(( self.nerwf.containsProcedure, RDFS.range,  self.xmlpo.Operation)) # a functional step of the workflow
        g.add(( self.nerwf.containsProcedure, RDFS.label,   Literal("containsProcedure", lang="en")))
        g.add(( self.nerwf.containsProcedure, RDFS.comment,   Literal("Specify an internal functionality of a workflow", lang="en")))
        
        g.add(( self.nerwf.applyTaggingFormat, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.applyTaggingFormat, RDF.type, OWL.FunctionalProperty) )
        g.add(( self.nerwf.applyTaggingFormat, RDFS.domain, self.edam.operation_0335 )) # Data formatting
        g.add(( self.nerwf.applyTaggingFormat, RDFS.range,  self.nerwf.TaggingFormat))
        g.add(( self.nerwf.applyTaggingFormat, RDFS.label,   Literal("applyTaggingFormat", lang="en")))
        g.add(( self.nerwf.applyTaggingFormat, RDFS.comment,   Literal("Sets up a chunck style formatting (IO, IOB, BILOU, BIOES, etc)", lang="en")))
        
        g.add(( self.nerwf.generatesDataset, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.generatesDataset, RDFS.domain, self.edam.operation_0004 )) 
        g.add(( self.nerwf.generatesDataset, RDFS.range,  self.xmlpo.Dataset))
        g.add(( self.nerwf.generatesDataset, RDFS.label,   Literal("generatesDataset", lang="en")))
        g.add(( self.nerwf.generatesDataset, RDFS.comment,   Literal("Generation of a dataset as a product of an operation execution", lang="en")))
        
        g.add(( self.nerwf.generatesModel, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.generatesModel, RDFS.domain, self.edam.operation_0335 )) 
        g.add(( self.nerwf.generatesModel, RDFS.range,  self.mesh.D000098412 ))
        g.add(( self.nerwf.generatesModel, RDFS.label,   Literal("generatesModel", lang="en")))
        g.add(( self.nerwf.generatesModel, RDFS.comment,   Literal("Generation of a predictive model as a product of an operation execution", lang="en")))
        
        g.add(( self.nerwf.useInputData, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.useInputData, RDFS.domain, self.xmlpo.Operation )) 
        range_union = BNode()
        Collection(g, range_union, [ self.xmlpo.Dataset, self.xmlpo.Data ])
        g.add(( self.nerwf.useInputData, RDFS.range,  range_union ))
        g.add(( self.nerwf.useInputData, RDFS.label,   Literal("useInputData", lang="en")))
        g.add(( self.nerwf.useInputData, RDFS.comment,   Literal("Correlates operation with some declared input data", lang="en")))
        
        g.add(( self.nerwf.hasScore, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.hasScore, RDFS.domain, self.xmlpo.ModelEvaluationCharacteristic )) 
        g.add(( self.nerwf.hasScore, RDFS.range,  self.nerwf.NEREvaluationMeasure ))
        g.add(( self.nerwf.hasScore, RDFS.label,   Literal("hasScore", lang="en")))
        g.add(( self.nerwf.hasScore, RDFS.comment,   Literal("Correlates the model evaluation settings with the scores", lang="en")))
        
        g.add(( self.nerwf.predictedByModel, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.predictedByModel, RDFS.domain, self.nerwf.SummaryPrediction )) 
        g.add(( self.nerwf.predictedByModel, RDFS.range, self.mesh.D000098412 ))
        g.add(( self.nerwf.predictedByModel, RDFS.label,   Literal("predictedByModel", lang="en")))
        g.add(( self.nerwf.predictedByModel, RDFS.comment,   Literal("Generation of a predictive model as a product of an operation execution", lang="en")))
        
        g.add(( self.nerwf.hasSummaryPrediction, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.hasSummaryPrediction, RDFS.domain, self.edam.operation_0335 )) 
        g.add(( self.nerwf.hasSummaryPrediction, RDFS.range, self.nerwf.SummaryPrediction ))
        g.add(( self.nerwf.hasSummaryPrediction, RDFS.label,   Literal("hasSummaryPrediction", lang="en")))
        g.add(( self.nerwf.hasSummaryPrediction, RDFS.comment,   Literal("Generation of a predictive model as a product of an operation execution", lang="en")))
        
        g.add(( self.nerwf.belongsToEntity, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.belongsToEntity, RDF.type, OWL.FunctionalProperty) )
        g.add(( self.nerwf.belongsToEntity, RDFS.domain, self.nerwf.NEREvaluationMeasure )) 
        g.add(( self.nerwf.belongsToEntity, RDFS.range,  self.nero.NamedEntity ))
        g.add(( self.nerwf.belongsToEntity, RDFS.label,   Literal("belongsToEntity", lang="en")))
        g.add(( self.nerwf.belongsToEntity, RDFS.comment,   Literal("Specify the name of the entity related to the score instance", lang="en")))
        
        # --------- Datatype Properties
        g.add(( self.nerwf.hasReplicateNumber, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.hasReplicateNumber, RDFS.domain, OWL.Class )) 
        g.add(( self.nerwf.hasReplicateNumber, RDFS.range, RDFS.Literal ))
        g.add(( self.nerwf.hasReplicateNumber, RDFS.label,   Literal("hasReplicateNumber", lang="en")))
        g.add(( self.nerwf.hasReplicateNumber, RDFS.comment,   Literal("The replicate index of the asset (model or other output)", lang="en")))
        
        g.add(( self.nerwf.isAggregatedValue, RDF.type, OWL.DatatypeProperty) )
        domain_union = BNode()
        Collection(g, domain_union, [ self.nerwf.SummaryPrediction, self.nerwf.NEREvaluationMeasure ])
        g.add(( self.nerwf.isAggregatedValue, RDFS.domain, domain_union )) 
        g.add(( self.nerwf.isAggregatedValue, RDFS.range,  RDFS.Literal ))
        g.add(( self.nerwf.isAggregatedValue, RDFS.label,   Literal("isAggregatedValue", lang="en")))
        g.add(( self.nerwf.isAggregatedValue, RDFS.comment, Literal("Describes whether the value of the score instance is a result of an statistical aggregation function (min, max, mean, etc)", lang="en")))
        
        g.add(( self.nerwf.aggregatedByStatsFunction, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.aggregatedByStatsFunction, RDFS.domain, self.nerwf.NEREvaluationMeasure )) 
        g.add(( self.nerwf.aggregatedByStatsFunction, RDFS.range, RDFS.Literal ))
        g.add(( self.nerwf.aggregatedByStatsFunction, RDFS.label,   Literal("aggregatedByStatsFunction", lang="en")))
        g.add(( self.nerwf.aggregatedByStatsFunction, RDFS.comment,   Literal("Describes statistical function used to aggregate the model replicate values of evaluation metrics", lang="en")))
        
        g.add(( self.nerwf.reportLevel, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.reportLevel, RDFS.domain, self.xmlpo.ModelEvaluationCharacteristic )) 
        g.add(( self.nerwf.reportLevel, RDFS.range, RDFS.Literal ))
        g.add(( self.nerwf.reportLevel, RDFS.label,   Literal("reportLevel", lang="en")))
        g.add(( self.nerwf.reportLevel, RDFS.comment,   Literal("Describes the level that the score was computed (word or token)", lang="en")))
        
        g.add(( self.nerwf.underContext, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.underContext, RDFS.domain, self.xmlpo.ModelEvaluationCharacteristic )) 
        g.add(( self.nerwf.underContext, RDFS.range, RDFS.Literal ))
        g.add(( self.nerwf.underContext, RDFS.label,   Literal("underContext", lang="en")))
        g.add(( self.nerwf.underContext, RDFS.comment,   Literal("Describes stage of the workflow in which the scores were computed", lang="en")))
        
        g.add(( self.nerwf.hasPredictedItemsCount, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.hasPredictedItemsCount, RDFS.domain, self.nerwf.SummaryPrediction )) 
        g.add(( self.nerwf.hasPredictedItemsCount, RDFS.range, RDFS.Literal ))
        g.add(( self.nerwf.hasPredictedItemsCount, RDFS.label,   Literal("hasPredictedItemsCount", lang="en")))
        g.add(( self.nerwf.hasPredictedItemsCount, RDFS.comment,   Literal("Number of words predicted to belong to certain entity", lang="en")))
        
        path = os.path.join( self.out, f'nerml_ontology.xml' )
        g.serialize( destination = path, format = 'xml' )
        txt = open( path ).read()
        txt = txt.replace('<rdf:RDF','<rdf:RDF xmlns="https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#"\nxml:base="https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#"\n')
        f = open( path, 'w')
        f.write(txt)
        f.close()

        self.graph = g

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

    def count_new_classes_properties(self):
        g = self.graph

        meta = { 'classes': 'owl:Class', 'object_properties': 'owl:ObjectProperty', 'data_properties': 'owl:DatatypeProperty' }
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

    def count_instances_per_class(self):
        g = self.graph

        res = {}
        q = '''
SELECT ?c (count( DISTINCT ?s ) as ?cnt )
WHERE {
    ?s rdf:type ?c .

    filter( (?c != owl:Class) && (?c != owl:ObjectProperty) && (?c != owl:DatatypeProperty) ) . 
}
group by ?c
        '''

        qres = g.query(q, initNs={'rdf': RDF, 'owl': OWL})
        for row in qres:
            print( f"{row.cnt} {row.c}" )
            res[row.c] = row.cnt
        
        return res

    def convert_ttl_to_owl(self):
        inpath = os.path.join( self.out, 'nerfairwf_onto_extension.ttl')
        opath = os.path.join( self.out, 'nerfairwf_onto_extension.owl')
        gr = rdflib.Graph()
        gr.parse(inpath)
        gr.serialize( destination=opath, format="xml")

    def _write_table(self, lines, opath):
        lines = list( map( lambda x: '\t'.join( [ str(el) for el in x] ), lines ))
        
        f = open( opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()
    
    def _load_onto_definitions(self, pcode):
        start = " _define_new_onto_elements"
        end = "self.graph = g"
        union_info = ''
        
        cnt_crossref = { }
        classes = set()
        prop_obj = set()
        prop_dat = set()
        d = {}
        flag = False
        f = open(pcode, "r")
        for line in f:
            l = line.strip()
            if( not l.startswith('#') ):
                if(l.find(end) != -1):
                    flag = False

                if( l.startswith('Collection') ):
                    aux = [ x.replace('self.','') for x in l.split(' [')[1].replace(']', '').split(', ') ]
                    union_info = aux

                if( flag and (l.find("g.add") != -1) ):
                    l = l.replace('g.add', '').replace('(', '').replace(')', '')
                    li = l.split(', ')

                    s = li[0].replace('self.','').strip()
                    p = li[1].replace('self.','').strip()
                    oaux = li[2].replace('self.','').strip()
                    o = li[2].replace('self.','').strip()
                    flag_literal = False

                    if( o.find('_union') != -1 ):
                        o = union_info
                    else:
                        if( o.find('Literal') != -1 ):
                            o = o.replace('Literal"', '').replace('"','').replace('")','').split(', ')[0]
                        o = [o]
                    print(s, p, o)
                    # Checking cross references to other ontologies
                    prefix = s.split('.')
                    if(len(prefix) > 1):
                        prefix = prefix[0]
                        if( prefix.islower() ):
                            if(not prefix in cnt_crossref ):
                                cnt_crossref[prefix] = { 'properties': 0, 'resources': 0 }
                            cnt_crossref[prefix]['resources'] += 1

                    prefix = p.split('.')
                    if(len(prefix) > 1):
                        prefix = prefix[0]
                        if( prefix.islower() ):
                            if(not prefix in cnt_crossref ):
                                cnt_crossref[prefix] = { 'properties': 0, 'resources': 0 }
                            cnt_crossref[prefix]['properties'] += 1

                    for el in o:
                        prefix = el.split('.')
                        if(len(prefix) > 1):
                            prefix = prefix[0]
                            if( prefix.islower() ):
                                if(not prefix in cnt_crossref ):
                                    cnt_crossref[prefix] = { 'properties': 0, 'resources': 0 }
                                cnt_crossref[prefix]['resources'] += 1

                    # organizing information
                    if( not s in d ):
                        d[s] = {}
                    d[s][p] = o

                    if( oaux == 'OWL.Class'):
                        classes.add(s)
                    if( oaux == 'OWL.ObjectProperty'):
                        prop_obj.add(s)
                    if( oaux == 'OWL.DatatypeProperty'):
                        prop_dat.add(s)
                        
                if(l.find(start) != -1):
                    flag = True

        f.close()

        # Organize ontology general metrics
        opath = os.path.join( self.out, 'table_general.tsv')
        lines = []
        lines.append( ['Feature', 'Quantity'] )
        lines.append( ['Axioms', 15] )
        lines.append( ['Classes', len( classes) ] )
        lines.append( ['Relationships', len(prop_obj) ] )
        lines.append( ['Attributes', len(prop_dat) ] )
        self._write_table(lines, opath)
        # axioms (subClassOf, sameAs, disjoint, functionalprop, transitiveprop)

        return d, cnt_crossref, classes, prop_obj, prop_dat

    def _get_prefixes(self, pcode):
        start = " _setup_namespaces"
        end = "self.graph = g"
        
        d = {}
        flag = False
        f = open(pcode, "r")
        for line in f:
            l = line.strip()
            if( not l.startswith('#') ):
                if(l.find(end) != -1):
                    flag = False

                if( flag and (l.find("Namespace") != -1) ):
                    l = l.replace('Namespace("', '').replace(')', '')
                    i = l.split(' = ')[0].replace('self.','')
                    url = l.split(' = ')[1]
                    d[i] = url
                
                if(l.find(start) != -1):
                    flag = True

        f.close()

        return d

    def _write_crossref_info(self, prefixes, cnt_crossref):
        # Cross-ref table information
        opath = os.path.join( self.out, 'crossref_metrics.tsv')
        lines = []
        lines.append( ['Prefix', 'URL', '# mention in properties', '# mention in resources'] )
        for pr in prefixes:
            url = prefixes[pr]
            if( pr in cnt_crossref ):
                np = cnt_crossref[pr]['properties']
                nr = cnt_crossref[pr]['resources']
                lines.append( [pr, url, np, nr] )
        self._write_table(lines, opath)

    def _write_classes_info(self, d, classes):
        props = ['RDFS.label', 'RDFS.comment', 'RDFS.subClassOf']
        opath = os.path.join( self.out, 'classes_information.tsv')
        lines = []
        lines.append( ['Class ID', 'Label', 'Description', 'Parent class'] )
        for el in classes:
            l = [el]
            for p in props:
                try:
                    l.append( ', '.join( d[el][p] ) )
                except:
                    l.append( '-' )
            lines.append( l )

        self._write_table(lines, opath)

    def _write_properties_info(self, d, prop_obj, prop_dat):
        props = ['RDFS.label', 'RDFS.comment', 'RDFS.domain', 'RDFS.range', 'OWL.inverseOf']
        opath = os.path.join( self.out, 'properties_information.tsv')
        lines = []
        lines.append( ['Property ID', 'Type', 'Label', 'Description', 'Domain', 'Range', 'Inverse Property'] )
        items = { 'Relationship': prop_obj, 'Attribute': prop_dat }
        for it in items:
            ids = items[it]
            for el in ids:
                l = [el, it]
                for p in props:
                    try:
                        l.append( ', '.join( d[el][p] ) )
                    except:
                        l.append( '-' )
                lines.append( l )

        self._write_table(lines, opath)

    def organize_onto_info_in_supp_tables(self):
        pcode = '/aloy/home/ymartins/match_clinical_trial/ner_subproj/modules/semantic_description.py'

        prefixes = self._get_prefixes(pcode)
        d, cnt_crossref, classes, prop_obj, prop_dat = self._load_onto_definitions(pcode)
        print(prefixes)
        self._write_crossref_info(prefixes, cnt_crossref)
        self._write_classes_info(d, classes)
        self._write_properties_info(d, prop_obj, prop_dat)

    def test_explanation_consistency_tec(self):
        inpath = os.path.join( self.out, 'all_nerfair_graph.xml')
        inpath = os.path.join( self.out, 'complete_nerml_ontology.xml' )
        inpath = '/aloy/home/ymartins/match_clinical_trial/out_eda_semantic/data/experiment_graph_biobert-bc5cdr-hypersearch.xml'

        onto = owlready2.get_ontology( inpath ).load()
        with onto: owlready2.sync_reasoner()
        opath = os.path.join( self.out, "test_onto_tec.owl")
        onto.save( opath )

    def execute_humanBased_queries(self):
        cq = "what are the f1-score values aggregated by max per model in the test context?"
        llmq = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX nf: <https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#>
PREFIX xhani: <https://w3id.org/ontouml-models/model/xhani2023xmlpo/>

SELECT ?f1ScoreValue
WHERE {
  ?evaluation nf:hasScore ?score .
  ?score rdf:type nf:NEREvaluationMeasure .
  ?score rdfs:label "f1-score" .
  ?score nf:hasValue ?f1ScoreValue .
  ?score nf:aggregatedByStatsFunction "max" .
  ?score nf:underContext "Test" .
  
}
"""
        prefixes = '''
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX nf: <https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#>
PREFIX xhani: <https://w3id.org/ontouml-models/model/xhani2023xmlpo/>
SELECT *
WHERE {
  ?workflow nf:containsProcedure ?operation .
  ?operation nf:generatesModel ?model .
  ?operation nf:describedBy ?evaluation .
  
  ?evaluation nf:hasScore ?score .
  ?evaluation nf:underContext "test" .
  ?score rdf:type nf:NEREvaluationMeasure .
  ?score nf:fromEvaluationMetric  stato:0000628 .
  ?score nf:hasValue ?f1ScoreValue .
  ?score nf:aggregatedByStatsFunction "max" .
  ?score nf:belongsToEntity ?ent .
  ?ent rdfs:label ?entity .
  
}
limit 4

Retrieve the number of models and datasets by experiment
'''        
        hq = """
SELECT ?level ?technique ?entity ?f1ScoreValue
WHERE {

  ?evaluation nf:hasScore ?score .
  ?evaluation nf:underContext ?ctx .
  ?evaluation nf:reportLevel ?level .
  ?evaluation xhani:MLEvaluationTechniqueName ?technique .

  ?score rdf:type nf:NEREvaluationMeasure .
  ?score nf:fromEvaluationMetric  stato:0000628 .
  ?score vcard:hasValue ?f1ScoreValue .
  ?score nf:aggregatedByStatsFunction ?agg .
  ?score nf:belongsToEntity ?ent .
  ?ent rdfs:label ?entity .
  
  filter(regex(?agg, "max", "i" )) .
}
"""

        inpath = os.path.join( self.out, 'all_nerfair_graph.ttl')
        
        gr = rdflib.Graph()
        gr.parse(inpath)

        vcard = Namespace("http://www.w3.org/2006/vcard/ns#")
        xhani = Namespace("https://w3id.org/ontouml-models/model/xhani2023xmlpo/")
        nf = Namespace("https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#")
        stato = Namespace("http://purl.obolibrary.org/obo/STATO_")
        nero = Namespace("http://www.cs.man.ac.uk/~stevensr/ontology/ner.owl#")
        
        qres = gr.query( hq, initNs = { 'rdf': RDF, 'rdfs': RDFS, 'xhani': xhani, 'nf': nf, 'stato': stato, 'vcard': vcard, "nero": nero } )
        #qres = gr.query( hq )
        for row in qres:
            print( row )


    def check_llm_queries(self):
        dat = {}

        cqs = [
            "Get the distinct names of named entities used in each experiment",
            "Retrieve the number of models and datasets by experiment",
            "Get the distinct evaluation metrics used to evaluate the models",
            "Get the distinct statistical functions used to aggregate the evaluation metrics",
            "Get the distinct evaluation techniques used in the experiments",
            "Retrieve the name and value of the hyperparameters used by each model",
            "what is the number of features and instances of the largest dataset",
            "which evaluation technique is associated to the highest mcc values",
            "For each experiment, retrieve the evaluation technique and the level that obtained the highest mcc values for each entity",
            "For each level, technique and entity, retrieve the f1-score values aggregated by max in the test context?"
        ]

        llms = { 
            'google': ChatGoogleGenerativeAI( model = self.google_model, temperature=0 ), 
            'llama': ChatOllama( model = self.llama_model, temperature=0 ) 
        }
        llms = { 
            'google': ["gemini-2.0-flash", "gemini-2.5-flash"], 
            'llama': ["llama3.2","deepseek-r1"]
        }

        inpath = os.path.join( self.out, 'all_nerfair_graph.ttl')
        graph = RdfGraph( source_file = inpath, standard='rdf')
        graph.load_schema()

        for provider in llms:
            print('--------> model type: ', provider)
            dat[provider] = {}
            models = llms[provider]

            for m in models:
                dat[provider][m] = {}
                if(provider == 'google'):
                    llm = ChatGoogleGenerativeAI( model = m, temperature=0 )
                if(provider == 'llama'):
                    llm = ChatOllama( model = self.llama_model, temperature=0 ) 
                chain = GraphSparqlQAChain.from_llm( llm, graph=graph, verbose=True, allow_dangerous_requests=True )

                for q in cqs:
                    print('\t question: ', q)
                    try:
                        result = chain.invoke( q )
                        #dat.append( { 'q': q, 'result': result } )
                        dat[provider][m][q] = result
                    except:
                        dat[provider][m][q] = "sparql syntax error"
                    time.sleep(60)

        opath = os.path.join( self.out, 'llm_query_results.json')
        json.dump( dat, open(opath, 'w') )

        hqs = [
            """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX nf: <https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#>
PREFIX onto: <https://w3id.org/ontouml-models/model/xhani2023xmlpo/>

SELECT DISTINCT ?experiment ?entityName
WHERE {
  ?experiment rdf:type nf:NLPExperiment .
  ?experiment onto:hasOutput ?dataset .
  ?dataset onto:datasetHasData ?namedEntity .
  ?namedEntity rdf:type <http://www.cs.man.ac.uk/~stevensr/ontology/ner.owl#NamedEntity> .
  ?namedEntity rdfs:label ?entityName .
}

            """
        ]

    def parse_llm_queries_result(self):
        path = os.path.join(self.out, 'stdout_sparql_llm-multiple_round2.txt')

        dat = {}
        flag = False
        i = 0
        lines = open(path, 'r').read().split('\n')[:-1]
        for line in lines:
            if(line.find('model type') != -1):
                model = line.split(':')[1].strip()
                dat[model] = {}

            elif(line.find('question:') != -1):
                cq = line.split(':')[1].replace(':','').strip()
                dat[model][cq] = { 'query': '', 'resq': '' }
                flag = False

            elif(line.find('Full Context:') != -1):
                content = lines[i+1].replace('[32;1m[1;3m','').replace('[0m','')
                try:
                    dat[model][cq]['resq'] = eval(content)
                except:
                    dat[model][cq]['resq'] = content

                flag = False

            elif( flag ):
                dat[model][cq]['query'] += line.replace('[32;1m[1;3m','').replace('[0m','')+' '

            elif(line.find('Generated SPARQL:') != -1):
                flag = True

            i += 1

        opath = os.path.join( self.out, 'parsed_llm_results.json')
        json.dump( dat, open(opath, 'w') )

        lines = []
        lines.append( ['model', 'cq', 'query', 'length_results'] )
        for m in dat:
            for cq in dat[m]:
                q = dat[m][cq]['query']
                nr = len( dat[m][cq]['resq'] )

                lines.append( [m, cq, q, nr ] )
        lines = list( map( lambda x: '\t'.join( [ str(el) for el in x] ), lines ))
        opath = os.path.join( self.out, 'table_llm_results.tsv')
        f = open(opath, 'w')
        f.write( '\n'.join(lines)+'\n' )
        f.close()

    def analysis_llm_queries(self):
        path = os.path.join( self.out, 'parsed_llm_results.json')
        d = json.load( open( path ) )

        ng = len( list( filter( lambda x: len(d['google'][x]['resq'])>0, d['google'] )) ) # 5
        nl = len( list( filter( lambda x: len(d['google'][x]['resq'])>0, d['llama'] )) ) # 5

        qganswered = list( filter( lambda x: len(d['google'][x]['resq'])>0, d['google'] ))
        qlanswered = list( filter( lambda x: len(d['google'][x]['resq'])>0, d['llama'] ))

        exl = d['llama']['Retrieve the name and value of the hyperparameters used by each model']['query']
        exg = d['google']['Retrieve the name and value of the hyperparameters used by each model']['query']

    def run(self):
        #self._define_new_onto_elements()
        #self.organize_onto_info_in_supp_tables()
        
        #self.count_new_classes_properties()
        #self.count_instances_per_class()

        #self.convert_ttl_to_owl()
        
        #self.rerun_meta_enrichment()
        #self.load_graphs()
        self.check_llm_queries()
        self.parse_llm_queries_result()
        #self.analysis_llm_queries()

        #self.execute_humanBased_queries()

        #self.test_explanation_consistency_tec()

if( __name__ == "__main__" ):
    odir = '../paper_files/out_eda_semantic'
    odir = '/aloy/home/ymartins/match_clinical_trial/out_eda_semantic'
    i = ExplorationSemanticResults( odir )
    i.run()
