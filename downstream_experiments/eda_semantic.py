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

#from owlready2 import *
import owlready2

from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD, BNode
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD, OWL
from rdflib.collection import Collection

os.environ["GOOGLE_API_KEY"] = "AIzaSyDV66WrLSzULt9PHN2gvqnx0qPmZsBHniI"
os.environ["OPENAI_API_KEY"] = "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"

from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.chains import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph

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

    def gen_id(self, prefix):
        _id = uuid4().hex
        return f'{prefix}_{_id}'

    def _setup_namespaces(self):
        g = self.graph

        self.nerwf = Namespace("https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfairwf_ontology.owl#")
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
        
        g.add(( self.xmlpo.Result, RDF.type, OWL.Class) )
        g.add(( self.xmlpo.Result, RDFS.label,   Literal("Result", lang="en")))
        g.add(( self.xmlpo.Result, RDFS.comment,   Literal("Refers to the experiment result", lang="en")))
        
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
        g.add(( self.nerwf.hasReplicateNumber, RDFS.range,  XSD.integer))
        g.add(( self.nerwf.hasReplicateNumber, RDFS.label,   Literal("hasReplicateNumber", lang="en")))
        g.add(( self.nerwf.hasReplicateNumber, RDFS.comment,   Literal("The replicate index of the asset (model or other output)", lang="en")))
        
        g.add(( self.nerwf.isAggregatedValue, RDF.type, OWL.DatatypeProperty) )
        domain_union = BNode()
        Collection(g, domain_union, [ self.nerwf.SummaryPrediction, self.nerwf.NEREvaluationMeasure ])
        g.add(( self.nerwf.isAggregatedValue, RDFS.domain, domain_union )) 
        g.add(( self.nerwf.isAggregatedValue, RDFS.range,  XSD.boolean))
        g.add(( self.nerwf.isAggregatedValue, RDFS.label,   Literal("isAggregatedValue", lang="en")))
        g.add(( self.nerwf.isAggregatedValue, RDFS.comment, Literal("Describes whether the value of the score instance is a result of an statistical aggregation function (min, max, mean, etc)", lang="en")))
        
        g.add(( self.nerwf.aggregatedByStatsFunction, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.aggregatedByStatsFunction, RDFS.domain, self.xmlpo.NEREvaluationMeasure )) 
        g.add(( self.nerwf.aggregatedByStatsFunction, RDFS.range,  XSD.string))
        g.add(( self.nerwf.aggregatedByStatsFunction, RDFS.label,   Literal("aggregatedByStatsFunction", lang="en")))
        g.add(( self.nerwf.aggregatedByStatsFunction, RDFS.comment,   Literal("Describes statistical function used to aggregate the model replicate values of evaluation metrics", lang="en")))
        
        g.add(( self.nerwf.reportLevel, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.reportLevel, RDFS.domain, self.xmlpo.ModelEvaluationCharacteristic )) 
        g.add(( self.nerwf.reportLevel, RDFS.range,  XSD.string))
        g.add(( self.nerwf.reportLevel, RDFS.label,   Literal("reportLevel", lang="en")))
        g.add(( self.nerwf.reportLevel, RDFS.comment,   Literal("Describes the level that the score was computed (word or token)", lang="en")))
        
        g.add(( self.nerwf.underContext, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.underContext, RDFS.domain, self.xmlpo.ModelEvaluationCharacteristic )) 
        g.add(( self.nerwf.underContext, RDFS.range,  XSD.string))
        g.add(( self.nerwf.underContext, RDFS.label,   Literal("underContext", lang="en")))
        g.add(( self.nerwf.underContext, RDFS.comment,   Literal("Describes stage of the workflow in which the scores were computed", lang="en")))
        
        g.add(( self.nerwf.hasPredictedItemsCount, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.hasPredictedItemsCount, RDFS.domain, self.nerwf.SummaryPrediction )) 
        g.add(( self.nerwf.hasPredictedItemsCount, RDFS.range,  XSD.integer))
        g.add(( self.nerwf.hasPredictedItemsCount, RDFS.label,   Literal("hasPredictedItemsCount", lang="en")))
        g.add(( self.nerwf.hasPredictedItemsCount, RDFS.comment,   Literal("Number of words predicted to belong to certain entity", lang="en")))
        
        path = os.path.join( self.out, f'complete_nerml_ontology.xml' )
        g.serialize( destination = path, format = 'xml' )

        self.graph = g

    def load_graphs(self):
        g = self.graph

        indir = os.path.join(self.out, 'data')
        for f in os.listdir(indir):
            path = os.path.join(indir, f)
            g.parse(path)

        opath = os.path.join( self.out, 'all_nerfair_graph.rdf')
        g.serialize( destination=opath, format="xml")

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
        inpath = os.path.join( self.out, 'complete_nerml_ontology.xml' )
        onto = get_ontology( inpath ).load()

        with onto: sync_reasoner()
        opath = os.path.join( self.out, "test_onto_tec.owl")
        onto.save( opath )

    def check_llm_queries(self):
        inpath = os.path.join( self.out, 'all_nerfair_graph.rdf')

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        #llm = ChatOpenAI(temperature=0)
        graph = RdfGraph( source_file = inpath, standard='rdf')
        graph.load_schema()
        chain = GraphSparqlQAChain.from_llm( llm, graph=graph, verbose=True, allow_dangerous_requests=True )
        #chain.invoke( "How many organisms have drug resistance?" )
        chain.invoke( "what are the f1-score values aggregated by max per model in the test context?" )
        print(f"SPARQL query: {result['sparql_query']}")
        print(f"Final answer: {result['result']}")

    def run(self):
        #self._define_new_onto_elements()
        #self.organize_onto_info_in_supp_tables()
        
        self.load_graphs()
        #self.count_new_classes_properties()
        #self.count_instances_per_class()

        #self.convert_ttl_to_owl()
        self.check_llm_queries()

if( __name__ == "__main__" ):
    odir = '/aloy/home/ymartins/match_clinical_trial/out_eda_semantic'
    i = ExplorationSemanticResults( odir )
    i.run()
