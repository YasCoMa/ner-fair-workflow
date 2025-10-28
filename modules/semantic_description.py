import os
import sys
import json
import glob
import pandas as pd
from uuid import uuid4
from datasets import load_from_disk

from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD, BNode
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD, OWL
from rdflib.collection import Collection

import argparse
import logging

class SemanticDescription:
    
    def __init__(self):
        self._get_arguments()
        self._setup_out_folders()

        self.fready = os.path.join( self.logdir, f"{self.expid}-tasks_semantic_description.ready")
        self.ready = os.path.exists( self.fready )
        if(self.ready):
            self.logger.info("----------- Semantic description step skipped since it was already computed -----------")
            self.logger.info("----------- Semantic description step ended -----------")
        
        self.approach = None

        self.graph = Graph()

        self._get_info_config()
        self._setup_namespaces()

    def _get_arguments(self):
        parser = argparse.ArgumentParser(description='Semantic description')
        parser.add_argument('-execDir','--execution_path', help='Directory where the logs and history will be saved', required=True)
        parser.add_argument('-paramFile','--parameter_file', help='Running configuration file', required=False)
        
        args = parser.parse_args()
        
        with open( args.parameter_file, 'r' ) as g:
            self.config = json.load(g)

        execdir = args.execution_path
        self.logdir = os.path.join( execdir, "logs" )
        if( not os.path.exists(self.logdir) ):
            os.makedirs( self.logdir )

        self.expid = self.config["identifier"]
        logf = os.path.join( self.logdir, f"{self.expid}-semantic_description.log" )
        logging.basicConfig( filename=logf, encoding="utf-8", filemode="a", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
        self.logger = logging.getLogger('prediction')

        try:
            self.expid = self.config["identifier"]
            self.model_checkpoint = self.config["pretrained_model"]
            self.outDirRoot = self.config["outpath"]
            self.target_tags = json.load( open( self.config["target_tags"], 'r') )

            self.logger.info("----------- Semantic description step started -----------")
        except:
            raise Exception("Mandatory fields not found in config. file")

    def _setup_out_folders(self):
        task = 'ner'
        model_name = self.model_checkpoint.split("/")[-1]
        fout = '.'
        if self.outDirRoot is not None:
            fout = self.outDirRoot
        self.outDir = os.path.join(fout, f"{self.expid}-{model_name}-finetuned-{task}" )

        self.out = os.path.join( self.outDir, "experiment_metadata" )
        if( not os.path.exists(self.out) ):
            os.makedirs( self.out )

    def gen_id(self, prefix):
        _id = str(uuid4())[:8]
        return f'{prefix}_{_id}'

    def _get_info_config(self):
        config = self.config

        self.exp_identifier = config["identifier"]
        self.exp_metadata = {  }
        if( 'experiment_metadata' in config ):
            self.exp_metadata = config['experiment_metadata']
        
        _id = self.gen_id('exp')
        self.exp_metadata['id'] = _id
        if( 'name' not in self.exp_metadata):
            self.exp_metadata['name'] = f'NLP Experiment for named entity recognition - identifier {_id}'
        if( 'description' not in self.exp_metadata):
            self.exp_metadata['description'] = ''
        if( 'domain' not in self.exp_metadata):
            self.exp_metadata['domain'] = ''
        if( 'source' not in self.exp_metadata):
            self.exp_metadata['source'] = ''

        '''
        "experiment_metadata": { 
            "name": "NER for PICO entities - Augmentation dataset", 
            "domain": "NER for PICO entities in texts concerning clinical trials", 
            "description": "Finetuning of the biobert LLM model for NER task applied to identification of entities on pubmed abstracts regarding clinical trials. These entities are related to participants characterization, interventions, control vectors and outcomes." ,
            "source": "The original input data comprises 1011 pubmed abstracts annotated by human curators concerning entities related to clinical trials. The link for the data is https://github.com/sociocom/PICO-Corpus/tree/main/pico_corpus_brat_annotated_files"
            }
        }
        '''

    def _setup_namespaces(self):
        g = self.graph

        self.nerwf = Namespace("https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#")
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
        self.logger.info("[Semantic description step] Task (Defining ontology elements) started -----------")
        
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

        self.logger.info("[Semantic description step] Task (Defining ontology elements) ended -----------")

    def __describe_workflow(self):
        self.logger.info("[Semantic description step] Task (Describing workflow) started -----------")
        
        g = self.graph
        _id = self.gen_id('wf')
        g.add( ( self.nerwf[_id], RDF.type, self.xmlpo.workflow ) )
        g.add( ( self.nerwf[_id], RDFS.label, Literal( "NER FAIR workflow", lang="en" ) ) )
        g.add( ( self.nerwf[_id], DCTERMS.description, Literal( self.exp_metadata['description']) ) )
        
        subject_union = BNode()
        Collection(g, subject_union, [ self.edam.topic_0218, self.edam.topic_0769 ]) # Natural language processing, workflows
        g.add( ( self.nerwf[_id], DCTERMS.subject, subject_union ) )
        
        self.graph = g

        self.wfid = _id
        
        self.logger.info("[Semantic description step] Task (Describing workflow) ended -----------")

    def __remove_prefix_tags(self, target_tags, remove_duplicates=False):
        categories = list( map( lambda x: x[2:] if( x[:2].lower() in ['o-', 'b-', 'i-', 'e-', 's-', 'u-', 'l-']) else x, target_tags ) )
        
        if( remove_duplicates ):
            aux = []
            for j, c in enumerate(categories):
                if( c not in aux ):
                    aux.append(c)
            categories = aux
        return categories
        
    def _describe_target_entities(self, expid):
        g = self.graph

        target_tags = self.target_tags
        categories = self.__remove_prefix_tags(target_tags, remove_duplicates=True)
        ents = target_tags + categories
        mapp = {}
        for entity in ents:
            ent = self.gen_id('entity') 
            g.add( ( self.nerwf[ent], RDF.type, self.nero.NamedEntity ) )
            g.add( ( self.nerwf[ent], RDFS.label, Literal(entity) ) )
            g.add( ( self.nerwf[expid], self.nerwf.containsTargetEntity, self.nerwf[ent] ) )
            mapp[entity] = ent

        self.mapp_entity_id = mapp

        self.graph = g

        return mapp

    def _describe_experiment(self):
        self.logger.info("[Semantic description step] Task (Describing experiment) started -----------")
        
        g = self.graph

        _id = self.exp_metadata['id']
        g.add( ( self.nerwf[_id], RDF.type, self.nerwf.NLPExperiment ) )

        expname = f"NER Experiment {_id}"
        if( self.exp_metadata['name'] != "" ):
            expname = self.exp_metadata['name']
        g.add( ( self.nerwf[_id], RDFS.label, Literal( expname ) ) )

        if( self.exp_metadata['description'] != "" ):
            g.add( ( self.nerwf[_id], DCTERMS.description, Literal( self.exp_metadata['description']) ) )
            g.add( ( self.nerwf[_id], self.xmlpo.description, Literal( self.exp_metadata['description']) ) )
        
        if( self.exp_metadata['source'] != "" ):
            g.add( ( self.nerwf[_id], DCTERMS.source, Literal( self.exp_metadata['source']) ) )
        
        ecid = self.gen_id('expChar')
        g.add( ( self.nerwf[ecid], RDF.type, self.xmlpo.ExperimentCharacteristics ) )
        g.add( ( self.nerwf[ecid], self.xmlpo.ExperimentType, Literal("Named entity recognition", lang="en") ) )
        g.add( ( self.nerwf[ecid], self.xmlpo.ExperimentDomain, Literal(self.exp_metadata['domain']) ) )
        g.add( ( self.nerwf[ecid], self.xmlpo.ExperimentDescription, Literal(self.exp_metadata['description']) ) )
        g.add( ( self.nerwf[_id], self.nerwf.describedBy, self.nerwf[ecid] ) )

        self.__describe_workflow()
        g.add( ( self.nerwf[_id], self.nerwf.executedBy, self.nerwf[self.wfid] ) )

        self._describe_target_entities(_id)

        self.graph = g

        self.logger.info("[Semantic description step] Task (Describing experiment) ended -----------")
        
    def __describe_preproc_dataset(self):
        g = self.graph
        
        ds1 = None
        dataPreprocDir = os.path.join(self.outDir, "preprocessing", "dataset_train_valid_test_split_v0.1") # Transformers dataset utput from preproc step
        flag = os.path.isdir(dataPreprocDir)
        if( flag ):
            ds = load_from_disk(dataPreprocDir)

            ds1 = self.gen_id('wfdataset') 
            g.add( ( self.nerwf[ds1], RDF.type, self.xmlpo.LabeledDataset ) )
            g.add( ( self.nerwf[ds1], RDFS.label, Literal( "Dataset splitted into train, test, and validation", lang="en") ) )
            
            # Features
            fe = self.gen_id('feature') 
            g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
            g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature ID", lang="en") ) )
            g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
            datc = self.gen_id('featureCharacteristics') 
            g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
            g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("id", lang="en") ) )
            g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Original file identifier for the sentences", lang="en" ) ) )
            g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )

            fe = self.gen_id('feature') 
            g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
            g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature TOKENS", lang="en") ) )
            g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
            datc = self.gen_id('featureCharacteristics') 
            g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
            g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("tokens", lang="en") ) )
            g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Original sentence tokens", lang="en" ) ) )
            g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )
            
            fe = self.gen_id('feature') 
            g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
            g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature NER_TAGS", lang="en") ) )
            g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
            datc = self.gen_id('featureCharacteristics') 
            g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
            g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("ner_tags", lang="en") ) )
            g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Correct entity labels for each token in the sentences", lang="en" ) ) )
            g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )

            # Data splits
            dtrain = self.gen_id('wfdata')
            self.dtrain = dtrain
            g.add( ( self.nerwf[dtrain], RDF.type, self.xmlpo.TrainSet ) )
            g.add( ( self.nerwf[dtrain], RDFS.label, Literal( "Train data", lang="en") ) )
            g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasData, self.nerwf[dtrain] ) )
            datc = self.gen_id('dataCharacteristics') 
            g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.DatasetCharacteristic ) )
            g.add( ( self.nerwf[datc], self.xmlpo.NumberOfFeatures, Literal(3) ) )
            g.add( ( self.nerwf[datc], self.xmlpo.NumberOfInstances, Literal( len( ds['train'] ) ) ) )
            g.add( ( self.nerwf[dtrain], self.nerwf.describedBy, self.nerwf[datc] ) )

            dtest = self.gen_id('wfdata') 
            self.dtest = dtest
            g.add( ( self.nerwf[dtest], RDF.type, self.xmlpo.TestSet ) )
            g.add( ( self.nerwf[dtest], RDFS.label, Literal( "Test data", lang="en") ) )
            g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasData, self.nerwf[dtest] ) )
            datc = self.gen_id('dataCharacteristics') 
            g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.DatasetCharacteristic ) )
            g.add( ( self.nerwf[datc], self.xmlpo.NumberOfFeatures, Literal(3) ) )
            g.add( ( self.nerwf[datc], self.xmlpo.NumberOfInstances, Literal( len( ds['test'] ) ) ) )
            g.add( ( self.nerwf[dtest], self.nerwf.describedBy, self.nerwf[datc] ) )
            
            dvalid = self.gen_id('wfdata') 
            g.add( ( self.nerwf[dvalid], RDF.type, self.nerwf.ValidationSet ) )
            g.add( ( self.nerwf[dvalid], RDFS.label, Literal( "Validation data", lang="en") ) )
            g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasData, self.nerwf[dvalid] ) )
            datc = self.gen_id('dataCharacteristics') 
            g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.DatasetCharacteristic ) )
            g.add( ( self.nerwf[datc], self.xmlpo.NumberOfFeatures, Literal(3) ) )
            g.add( ( self.nerwf[datc], self.xmlpo.NumberOfInstances, Literal( len( ds['valid'] ) ) ) )
            g.add( ( self.nerwf[dvalid], self.nerwf.describedBy, self.nerwf[datc] ) )
            
        self.graph = g

        return flag, ds1

    def _describe_data_preprocessing(self):
        self.logger.info("[Semantic description step] Task (Describing preprocessed datasets) started -----------")
        
        g = self.graph
        
        flag, dataset_id = self.__describe_preproc_dataset()
        if(flag):
            proc1 = self.gen_id('wfoperation') # Data Handling => Parsing
            g.add( ( self.nerwf[proc1], RDF.type, self.edam.operation_1812 ) ) # Data parsing
            g.add( ( self.nerwf[proc1], RDFS.label, Literal("Process texts and convert annotations to coNLL format", lang="en") ) )
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )

            proc2 = self.gen_id('wfoperation') # Data Handling => Formatting
            g.add( ( self.nerwf[proc2], RDF.type, self.edam.operation_0335 ) )
            g.add( ( self.nerwf[proc2], RDFS.label, Literal("Format coNLL annotations into IOB format", lang="en") ) )
            g.add( ( self.nerwf[proc2], self.nerwf.applyTaggingFormat, self.nerwf.nlpformat_iob ) )
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc2] ) )
            
            preprocParamConfig = self.gen_id('paramSetting') 
            g.add( ( self.nerwf[preprocParamConfig], RDF.type, self.xmlpo.ParameterSettings ) )
            g.add( ( self.nerwf[preprocParamConfig], RDFS.label, Literal("Proprocessing parameter configuration", lang="en") ) )
            g.add( ( self.nerwf[proc2], self.xmlpo.HasParameter, self.nerwf[preprocParamConfig] ) )
            
            value = True
            if('eliminate_overlappings' in self.config):
                if( self.config["eliminate_overlappings"] in [True, False] ):
                    value = self.config["eliminate_overlappings"]
            param = self.gen_id('preprocparam') 
            g.add( ( self.nerwf[param], RDF.type, self.xmlpo.ParameterCharacteristic ) )
            g.add( ( self.nerwf[param], self.xmlpo.ParameterName, Literal( "eliminate entity annotation overlapping", lang="en") ) )
            g.add( ( self.nerwf[param], self.xmlpo.ParameterValue, Literal(value) ) )
            g.add( ( self.nerwf[preprocParamConfig], self.xmlpo.hasParameterCharacteristic, self.nerwf[param] ) )

            proc3 = self.gen_id('wfoperation') # DatasetSplit
            g.add( ( self.nerwf[proc3], RDF.type, self.xmlpo.DatasetSplit ) )
            g.add( ( self.nerwf[proc3], RDFS.label, Literal("Reorganize the data according to the train, test and validation sets", lang="en") ) )
            g.add( ( self.nerwf[proc3], self.nerwf.generatesDataset, self.nerwf[dataset_id] ) )
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc3] ) )
            
        self.graph = g

        self.logger.info("[Semantic description step] Task (Describing preprocessed datasets) ended -----------")
        
    def __check_model_details(self, train_id):
        g = self.graph

        self.models = {}
        
        hyperparams = {'per_device_train_batch_size': 8, 'per_device_eval_batch_size': 16, 'learning_rate': 2e-5, 'weight_decay': 0.01 }
        trout = os.path.join( self.outDir, "training" )
        flag = os.path.exists(trout)
        if(flag):
            model_files = [file for file in os.listdir(self.outDir) if file.startswith("model_")]
            if( len(model_files) > 0 ):
                do_optimization = False
                optimization_file = None
                if( 'do_hyperparameter_search' in self.config ):
                  do_optimization = self.config["do_hyperparameter_search"]
                if( 'hyperparameter_path' in self.config ):
                  optimization_file = self.config["hyperparameter_path"]
                  if( optimization_file == "" ):
                      optimization_file = None

                if( optimization_file is None ):
                    hyperparams  = pickle.load( open( optimization_file, 'rb' ))
                elif(do_optimization):
                    pf = glob.glob(os.path.join(trout, "*.pkl"))
                    if( len(pf) > 0 ):
                        hyperparams  = pickle.load( open(pf[0], 'rb' ))

                # Defining hyperparameters
                trainParamConfig = self.gen_id('paramSetting') 
                g.add( ( self.nerwf[trainParamConfig], RDF.type, self.xmlpo.ParameterSettings ) )
                g.add( ( self.nerwf[trainParamConfig], RDFS.label, Literal("Training hyper parameter configuration", lang="en") ) )
                g.add( ( self.nerwf[train_id], self.xmlpo.HasParameter, self.nerwf[trainParamConfig] ) )

                for hp in hyperparams:
                    value = hyperparams[hp]
                    param = self.gen_id('hyperparam') 
                    g.add( ( self.nerwf[param], RDF.type, self.xmlpo.ParameterCharacteristic ) )
                    g.add( ( self.nerwf[param], self.xmlpo.ParameterName, Literal(hp, lang="en") ) )
                    g.add( ( self.nerwf[param], self.xmlpo.ParameterValue, Literal(value) ) )
                    g.add( ( self.nerwf[trainParamConfig], self.xmlpo.hasParameterCharacteristic, self.nerwf[param] ) )

                # Defining seed
                self.seed = 42
                if( 'seed' in self.config ):
                    if( isinstance( self.config['seed'], int) ):
                        self.seed = self.config['seed']
                value = self.seed
                param = self.gen_id('trainparam') 
                g.add( ( self.nerwf[param], RDF.type, self.xmlpo.ParameterCharacteristic ) )
                g.add( ( self.nerwf[param], self.xmlpo.ParameterName, Literal("Fixed initial value for random number generator", lang="en") ) )
                g.add( ( self.nerwf[param], self.xmlpo.ParameterValue, Literal(value) ) )
                g.add( ( self.nerwf[trainParamConfig], self.xmlpo.hasParameterCharacteristic, self.nerwf[param] ) )

                self.optimization_metric = 'f1'
                if( 'optimization_metric' in self.config ):
                    if( self.config['optimization_metric'] in ['f1', 'precision', 'recall', 'accuracy'] ):
                        self.optimization_metric = self.config['optimization_metric']
                value = self.optimization_metric
                param = self.gen_id('trainparam') 
                g.add( ( self.nerwf[param], RDF.type, self.xmlpo.ParameterCharacteristic ) )
                g.add( ( self.nerwf[param], self.xmlpo.ParameterName, Literal("Optimization metric used for hyperparameter search", lang="en") ) )
                g.add( ( self.nerwf[param], self.xmlpo.ParameterValue, Literal(value) ) )
                g.add( ( self.nerwf[trainParamConfig], self.xmlpo.hasParameterCharacteristic, self.nerwf[param] ) )
                
                # Defining models
                rootmodel = self.gen_id('original_model')
                g.add( ( self.nerwf[rootmodel], RDF.type, self.mesh.D000098342 ) )
                g.add( ( self.nerwf[rootmodel], RDFS.label, Literal( f"Original model {self.model_checkpoint}", lang="en") ) )
                for i, model in enumerate(model_files):
                    finemodel = self.gen_id('finetuned_model')
                    self.models[finemodel] = model
                    g.add( ( self.nerwf[finemodel], RDF.type, self.mesh.D000098342 ) )
                    g.add( ( self.nerwf[finemodel], RDFS.label, Literal( f"Finetunned {model}", lang="en") ) )
                    g.add( ( self.nerwf[finemodel], self.nerwf.finetunedFrom, self.nerwf[rootmodel] ) )
                    

        self.graph = g

        return flag, self.models    

    def _integrate_model_evaluation_agg_results(self, eval_op_id, stage):
        folder = os.path.join( self.outDir, stage, 'summary_reports' )
        
        g = self.graph

        eval_links = { 'accuracy': self.stato["0000415"], 'precision': self.stato["0000416"], 'recall': self.stato["0000233"], 'aucroc': self.stato["0000608"], 'f1-score': self.stato["0000628"], 'mcc': self.stato["0000524"], 'kappa': self.stato["0000630"] }
        evaluation_modes = ['seqeval-default', 'seqeval-strict', 'sk-with-prefix', 'sk-without-prefix']
        levels = ['token', 'word']
        for mode in evaluation_modes:
            for level in levels:
                files = glob.glob( os.path.join(folder, f"{mode}_summary-report_*-l{level}.tsv") )
                if( len(files) > 0 ):
                    datc = self.gen_id('evalCharacteristics') 
                    g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.ModelEvaluationCharacteristic ) )
                    g.add( ( self.nerwf[datc], self.xmlpo.MLEvaluationTechniqueName, Literal( mode, lang="en" ) ) )
                    g.add( ( self.nerwf[datc], self.nerwf.reportLevel, Literal( level, lang="en" ) ) )
                    g.add( ( self.nerwf[datc], self.nerwf.underContext, Literal( stage, lang="en" ) ) )
                    g.add( ( self.nerwf[eval_op_id], self.nerwf.describedBy, self.nerwf[datc] ) )
                    for f in files:
                        df = pd.read_csv(f, sep='\t')
                        for i in df.index:
                            entity = df.loc[i, 'Entity']
                            evalMetric = df.loc[i, 'evaluation_metric']
                            statsMetric = df.loc[i, 'stats_agg_name']
                            value = df.loc[i, 'stats_agg_value']

                            ent = self.mapp_entity_id[entity]

                            score = self.gen_id('evalScore') 
                            g.add( ( self.nerwf[score], RDF.type, self.nerwf.NEREvaluationMeasure ) )
                            g.add( ( self.nerwf[score], self.nerwf.belongsToEntity, self.nerwf[ent] ) )
                            g.add( ( self.nerwf[score], self.nerwf.isAggregatedValue, Literal(True) ) )
                            g.add( ( self.nerwf[score], self.vcard.hasValue, Literal( value ) ) )
                            g.add( ( self.nerwf[score], self.nerwf.aggregatedByStatsFunction, Literal( statsMetric, lang="en" ) ) )
                            if( evalMetric in eval_links ):
                                g.add( ( eval_links[evalMetric], RDFS.label, Literal( evalMetric, lang="en") ) )
                                g.add( ( self.nerwf[score], self.nerwf.fromEvaluationMetric, eval_links[evalMetric] ) )
                            g.add( ( self.nerwf[datc], self.nerwf.hasScore, self.nerwf[score] ) )

        self.graph = g
    
    def _describe_training(self):
        self.logger.info("[Semantic description step] Task (Describing training results) started -----------")
        
        g = self.graph

        proc1 = self.gen_id('wfoperation')
        flag, models = self.__check_model_details(proc1)
        if(flag):
            g.add( ( self.nerwf[proc1], RDF.type, self.xmlpo.Train ) )
            g.add( ( self.nerwf[proc1], RDFS.label, Literal("Training step", lang="en") ) )
            g.add( ( self.nerwf[proc1], self.nerwf.useInputData, self.nerwf[self.dtrain] ) )
            
            self.approach = self.gen_id('mlapproach')
            g.add( ( self.nerwf[self.approach], RDF.type, self.xmlpo.ClassificationAlgorithm ) )
            g.add( ( self.nerwf[self.approach], RDFS.label, Literal("Deep Learning - Transformers", lang="en") ) )
            g.add( ( self.nerwf[proc1], self.stato["0000102"], self.nerwf[self.approach] ) ) # operation executes 
            
            for model_id in models:
                g.add( ( self.nerwf[self.approach], self.xmlpo.hasOutput, self.nerwf[model_id] ) )
                g.add( ( self.nerwf[proc1], self.nerwf.generatesModel, self.nerwf[model_id] ) )
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )

            proc1 = self.gen_id('wfoperation')
            g.add( ( self.nerwf[proc1], RDF.type, self.xmlpo.MLModelEvaluation ) )
            g.add( ( self.nerwf[proc1], RDFS.label, Literal("Training - model evaluation", lang="en") ) )
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )
            self._integrate_model_evaluation_agg_results(proc1, 'train')

        self.graph = g

        self.logger.info("[Semantic description step] Task (Describing training results) ended -----------")
        
    def _describe_test(self):
        self.logger.info("[Semantic description step] Task (Describing test results) started -----------")
        
        g = self.graph

        stage = 'test'
        pattern = os.path.join( self.outDir, stage, 'summary_reports', '*.tsv' )
        files = glob.glob(pattern)
        flag = ( len(files) > 0 )
        if(flag):
            proc1 = self.gen_id('wfoperation') 
            g.add( ( self.nerwf[proc1], RDF.type, self.xmlpo.Test ) )
            g.add( ( self.nerwf[proc1], RDFS.label, Literal("Test step", lang="en") ) )
            g.add( ( self.nerwf[proc1], self.nerwf.useInputData, self.nerwf[self.dtest] ) )
            
            approach = self.approach
            if( self.approach is None ):
                approach = self.gen_id('mlapproach')
                g.add( ( self.nerwf[approach], RDF.type, self.xmlpo.ClassificationAlgorithm ) )
                g.add( ( self.nerwf[approach], RDFS.label, Literal("Deep Learning - Transformers", lang="en") ) )
            g.add( ( self.nerwf[proc1], self.stato["0000102"], self.nerwf[approach] ) ) # operation executes 
            
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )
            
            proc1 = self.gen_id('wfoperation')
            g.add( ( self.nerwf[proc1], RDF.type, self.xmlpo.MLModelEvaluation ) )
            g.add( ( self.nerwf[proc1], RDFS.label, Literal("Test - model evaluation", lang="en") ) )
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )
            self._integrate_model_evaluation_agg_results(proc1, 'test')

        self.graph = g

        self.logger.info("[Semantic description step] Task (Describing test results) ended -----------")
        
    def __describe_out_prediction_dataset(self):
        g = self.graph

        ds1 = self.gen_id('wfdataset') 
        g.add( ( self.nerwf[ds1], RDF.type, self.xmlpo.LabeledDataset ) )
        g.add( ( self.nerwf[ds1], RDFS.label, Literal( "Dataset table containing predictions for the input set of text files", lang="en") ) )
        
        # Features
        fe = self.gen_id('feature') 
        g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
        g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature INPUT_FILE", lang="en") ) )
        g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
        datc = self.gen_id('featureCharacteristics') 
        g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("input_file", lang="en") ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Original file identifier took from text file name", lang="en" ) ) )
        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )

        fe = self.gen_id('feature') 
        g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
        g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature SCORE", lang="en") ) )
        g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
        datc = self.gen_id('featureCharacteristics') 
        g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("score", lang="en") ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Prediction score assigned by the model", lang="en" ) ) )
        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )
        
        fe = self.gen_id('feature') 
        g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
        g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature START", lang="en") ) )
        g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
        datc = self.gen_id('featureCharacteristics') 
        g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("start", lang="en") ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Start position where the entity span was identified", lang="en" ) ) )
        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )
        
        fe = self.gen_id('feature') 
        g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
        g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature END", lang="en") ) )
        g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
        datc = self.gen_id('featureCharacteristics') 
        g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("end", lang="en") ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "End position where the entity span was identified", lang="en" ) ) )
        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )
        
        fe = self.gen_id('feature') 
        g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
        g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature ENTITY_GROUP", lang="en") ) )
        g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
        datc = self.gen_id('featureCharacteristics') 
        g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("entity_group", lang="en") ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Entity predicted for the text span", lang="en" ) ) )
        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )
        
        fe = self.gen_id('feature') 
        g.add( ( self.nerwf[fe], RDF.type, self.xmlpo.Feature ) )
        g.add( ( self.nerwf[fe], RDFS.label, Literal( "Dataset feature WORD", lang="en") ) )
        g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasFeature, self.nerwf[fe] ) )
        datc = self.gen_id('featureCharacteristics') 
        g.add( ( self.nerwf[datc], RDF.type, self.xmlpo.FeatureCharacteristic ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("word", lang="en") ) )
        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "The piece of text that was applied to the entity", lang="en" ) ) )
        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )

        self.graph = g

        return ds1

    def __describe_summary_results_per_entity(self, procedure_id):
        stage = 'prediction' # change to train later
        
        folder = os.path.join( self.outDir, stage )
        
        g = self.graph

        for mid in self.models:
            mname = self.models[mid]
            path = os.path.join(folder, f"results_{m}.tsv") 
            df = pd.read_csv(path, sep='\t')
            entities = df.entity.unique()
            for e in entities:
                aux = df[ df.entity == e ]
                n = len(aux)
                npmids = len( set( [ x.split('_')[0] for x in aux.input_file ] ))

                sp = self.gen_id('sp') 
                g.add( ( self.nerwf[sp], RDF.type, self.nerwf.SummaryPrediction ) )
                g.add( ( self.nerwf[sp], RDFS.label, Literal( f"Prediction Summary for entity {e} - model {mname}", lang="en") ) )
                g.add( ( self.nerwf[sp], self.nerwf.belongsToEntity, Literal(e) ) )
                g.add( ( self.nerwf[sp], self.nerwf.predictedByModel, mid ) )
                g.add( ( self.nerwf[sp], self.nerwf.hasPredictedItemsCount, Literal(n) ) )
                g.add( ( self.nerwf[procedure_id], self.nerwf.hasSummaryPrediction, self.nerwf[sp] ) )

        self.graph = g

    def _describe_prediction(self):
        self.logger.info("[Semantic description step] Task (Describing prediction results) started -----------")
        
        g = self.graph

        stage = 'prediction'
        pattern = os.path.join( self.outDir, stage, 'result_*.tsv' )
        files = glob.glob(pattern)
        flag = ( len(files) > 0 )
        if(flag):
            proc1 = self.gen_id('wfoperation') # edam - Prediction and recognition
            g.add( ( self.nerwf[proc1], RDF.type, self.edam.operation_2423 ) ) # Prediction and recognition
            g.add( ( self.nerwf[proc1], RDFS.label, Literal("Prediction step on new data", lang="en") ) )
            
            # Declare input
            ds1 = self.gen_id('wfdataset') 
            g.add( ( self.nerwf[ds1], RDF.type, self.xmlpo.UnlabeledDataset ) )
            g.add( ( self.nerwf[ds1], RDFS.label, Literal( "Dataset composed of multiple text files", lang="en") ) )
            g.add( ( self.nerwf[proc1], self.nerwf.useInputData, self.nerwf[ds1] ) )
            
            # Declare output
            dsout = self.__describe_out_prediction_dataset()
            g.add( ( self.nerwf[proc1], self.nerwf.generatesDataset, self.nerwf[dsout] ) )
            self.__describe_summary_results_per_entity(proc1)

            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )

        self.graph = g

        self.logger.info("[Semantic description step] Task (Describing prediction results) ended -----------")
        
    def _export_graph(self):
        self.logger.info("[Semantic description step] Task (Exporting enrichment graph) started -----------")
        
        g = self.graph
        path = os.path.join( self.out, f'experiment_graph_{ self.exp_identifier }.ttl' )
        g.serialize( destination = path )
        path = os.path.join( self.out, f'experiment_graph_{ self.exp_identifier }.xml' )
        g.serialize( destination = path, format = 'xml' )

        txt = open( path ).read()
        txt = txt.replace('<rdf:RDF','<rdf:RDF xmlns="https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#"\nxml:base="https://raw.githubusercontent.com/YasCoMa/ner-fair-workflow/refs/heads/master/nerfair_onto_extension.owl#"\n')
        f = open( path, 'w')
        f.write(txt)
        f.close()

        self.logger.info("[Semantic description step] Task (Exporting enrichment graph) ended -----------")
        
    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Semantic description step ended -----------")
    
    def run(self):
        self._define_new_onto_elements()
        self._describe_experiment()
        self._describe_data_preprocessing()
        self._describe_training()
        self._describe_test()
        self._describe_prediction()
        self._export_graph()
        self._mark_as_done()

if( __name__ == "__main__" ):
    i = SemanticDescription( )
    if( not i.ready ):
        i.run()