import os
import sys
import json
from uuid import uuid4
from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD
from rdflib.namespace import DCTERMS, FOAF, PROV, RDFS, XSD, OWL

class SemanticDescription:
    
    def __init__(self, fout):
        self.config = json.load( open('/aloy/home/ymartins/match_clinical_trial/experiments/config_biobert.json','r') )
        self.outDir = self.config["outpath"]
        
        self.exp_metadata = {  }
        if( 'experiment_metadata' in config ):
            self.exp_metadata = config['experiment_metadata']
        
        _id = self.gen_id('exp')
        self.exp_metadata['id'] = _id
        if( 'name' not in self.exp_metadata):
            self.exp_metadata['name'] = f'NLP Experiment for namd entity recognition - identifier {_id}'
        if( 'description' not in self.exp_metadata):
            self.exp_metadata['description'] = ''
        if( 'domain' not in self.exp_metadata):
            self.exp_metadata['domain'] = ''
        if( 'source' not in self.exp_metadata):
            self.exp_metadata['source'] = ''
        
    
    def gen_id(self, prefix):
        _id = uuid4().hex
        return f'{prefix}_{_id}'

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

        self.graph = g
        
    def __define_tagging_format_instances(self):
    	g.add( ( self.nerwf.nlpformat_io, RDF.type, self.nerwf.TaggingFormat ) )
        g.add( ( self.nerwf.nlpformat_io, RDFS.label, Literal( "IO" ) ) )
        
        g.add( ( self.nerwf.nlpformat_iob, RDF.type, self.nerwf.TaggingFormat ) )
        g.add( ( self.nerwf.nlpformat_iob, RDFS.label, Literal( "IOB" ) ) )
        
        g.add( ( self.nerwf.nlpformat_bioes, RDF.type, self.nerwf.TaggingFormat ) )
        g.add( ( self.nerwf.nlpformat_bioes, RDFS.label, Literal( "BIOES" ) ) )
        
    def _define_new_onto_elements(self):
        g = self.graph
        
        # Defining specific types of Experiment
        g.add( ( self.nerwf.MLExperiment, RDFS.subClassOf, self.xmlpo.Experiment ) )
        g.add(( self.nerwf.MLExperiment, RDFS.label,   Literal("MLExperiment", lang="en")))
        
        g.add( ( self.nerwf.NLPExperiment, RDFS.subClassOf, self.nerwf.MLExperiment ) )
        g.add(( self.nerwf.NLPExperiment, RDFS.label,   Literal("NLPExperiment", lang="en")))
        
        g.add( ( self.nerwf.NEREvaluationMeasure, RDFS.subClassOf, self.xmlpo.ClassPredictionEvaluationMeasure ) )
        g.add(( self.nerwf.NEREvaluationMeasure, RDFS.label,   Literal("NEREvaluationMeasure", lang="en")))
        
        g.add( ( self.nerwf.ValidationSet, RDFS.subClassOf, self.xmlpo.PreprocessedData ) )
        g.add(( self.nerwf.ValidationSet, RDFS.label,   Literal("ValidationSet", lang="en")))
        
        g.add( ( self.nerwf.TaggingFormat, RDFS.subClassOf, self.edam.format_3862 ) ) # NLP annotation format class from edam (format branch)
        g.add(( self.nerwf.TaggingFormat, RDFS.label,   Literal("TaggingFormat", lang="en")))
        self.__define_tagging_format_instances()

        # Defining MLModel as a sub class of the general Model class
        g.add( ( self.xmlpo.MLModel, RDFS.subClassOf, self.ncit.C43383 ) ) 
        # Defining Predictive Learning Models model as a sub class of the MLModel class
        g.add( ( self.mesh.D000098412, RDFS.subClassOf, self.xmlpo.MLModel ) ) 
        # Defining LLM model as a sub class of the Predictive Learning Models class
        g.add( ( self.mesh.D000098342, RDFS.subClassOf, self.mesh.D000098412 ) ) 
        # Defining that Operation in EDAM has the same Meaning as in xmlpo
        g.add(( self.xmlpo.Operation, OWL.sameAs,  self.edam.operation_0004 )) # stato statistic
        
        # --------- Object Properties
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
        
        g.add(( self.nerwf.fromEvaluationMetric, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.fromEvaluationMetric, RDFS.domain, self.nerwf.NEREvaluationMeasure )) # Model
        g.add(( self.nerwf.fromEvaluationMetric, RDFS.range,  self.stato.0000039)) # stato statistic
        g.add(( self.nerwf.fromEvaluationMetric, RDFS.label,   Literal("fromEvaluationMetric", lang="en")))
        g.add(( self.nerwf.fromEvaluationMetric, RDFS.comment,   Literal("It can be used to specify the model parameters (hyper parameters)", lang="en")))
        
        g.add(( self.nerwf.containsProcedure, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.containsProcedure, RDFS.domain, self.xmlpo.workflow )) # Model
        g.add(( self.nerwf.containsProcedure, RDFS.range,  self.xmlpo.Operation)) # stato statistic
        g.add(( self.nerwf.containsProcedure, RDFS.label,   Literal("containsProcedure", lang="en")))
        g.add(( self.nerwf.containsProcedure, RDFS.comment,   Literal("Specify an internal functionality of a workflow", lang="en")))
        
        g.add(( self.nerwf.applyTaggingFormat, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.applyTaggingFormat, RDFS.domain, self.edam.operation_0335 )) 
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
        g.add(( self.nerwf.useInputData, RDFS.range,  self.xmlpo.Data ))
        g.add(( self.nerwf.useInputData, RDFS.label,   Literal("useInputData", lang="en")))
        g.add(( self.nerwf.useInputData, RDFS.comment,   Literal("Correlates operation with some declared input data", lang="en")))
        
        g.add(( self.nerwf.hasScore, RDF.type, OWL.ObjectProperty) )
        g.add(( self.nerwf.hasScore, RDFS.domain, self.xmlpo.ModelEvaluationCharacteristic )) 
        g.add(( self.nerwf.hasScore, RDFS.range,  self.nerwf.NEREvaluationMeasure ))
        g.add(( self.nerwf.hasScore, RDFS.label,   Literal("hasScore", lang="en")))
        g.add(( self.nerwf.hasScore, RDFS.comment,   Literal("Correlates the model evaluation settings with the scores", lang="en")))
        
        # --------- Datatype Properties
        g.add(( self.nerwf.hasReplicateNumber, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.hasReplicateNumber, RDFS.domain, self.OWL.Class )) 
        g.add(( self.nerwf.hasReplicateNumber, RDFS.range,  XSD.integer))
        g.add(( self.nerwf.hasReplicateNumber, RDFS.label,   Literal("hasReplicateNumber", lang="en")))
        g.add(( self.nerwf.hasReplicateNumber, RDFS.comment,   Literal("The replicate index of the asset (model or other output)", lang="en")))
        
        g.add(( self.nerwf.belongsToEntity, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.belongsToEntity, RDFS.domain, self.nerwf.NEREvaluationMeasure )) 
        g.add(( self.nerwf.belongsToEntity, RDFS.range,  XSD.string))
        g.add(( self.nerwf.belongsToEntity, RDFS.label,   Literal("belongsToEntity", lang="en")))
        g.add(( self.nerwf.belongsToEntity, RDFS.comment,   Literal("Specify the name of the entity related to the score instance", lang="en")))
        
        g.add(( self.nerwf.isAggregatedValue, RDF.type, OWL.DatatypeProperty) )
        g.add(( self.nerwf.isAggregatedValue, RDFS.domain, self.nerwf.NEREvaluationMeasure )) 
        g.add(( self.nerwf.isAggregatedValue, RDFS.range,  XSD.boolean))
        g.add(( self.nerwf.isAggregatedValue, RDFS.label,   Literal("isAggregatedValue", lang="en")))
        g.add(( self.nerwf.isAggregatedValue, RDFS.comment,   Literal("Describes whether the value of the score instance is a result of an statistical aggregation function (min, max, mean, etc)", lang="en")))
        
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
        
        
        self.graph = g

    def __describe_workflow(self);
        g = self.graph
        _id = self.gen_id('wf')
        g.add( ( self.nerwf[_id], RDF.type, self.xmlpo.workflow ) )
        g.add( ( self.nerwf[_id], RDFS.label, Literal( "NER FAIR workflow", lang="en" ) ) )
        g.add( ( self.nerwf[_id], DCTERMS.description, Literal( self.exp_metadata['description']) ) )
        subject_union = BNode()
		g.add((subject_union, RDF.type, OWL.Class))
		g.add((subject_union, OWL.unionOf, RDF.collection([ self.edam.topic_0218, self.edam.topic_0769 ]) ) )
        g.add( ( self.nerwf[_id], DCTERMS.subject, subject_union ) )
        
        self.graph = g

        self.wfid = _id

    def _describe_experiment(self):
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
        g.add( ( self.nerwf[ecid], self.xmlpo.ExperimentType, "Named entity recognition" ) )
        g.add( ( self.nerwf[ecid], self.xmlpo.ExperimentDomain, self.exp_metadata['domain'] ) )
        g.add( ( self.nerwf[ecid], self.xmlpo.ExperimentDescription, self.exp_metadata['description'] ) )
		g.add( ( self.nerwf[_id], self.nerwf.describedBy, self.nerwf[ecid] ) )

        self.__describe_workflow()
        g.add( ( self.nerwf[_id], self.nerwf.executedBy, self.nerwf[self.wfid] ) )

        self.graph = g

    def __describe_preproc_dataset(self):
        g = self.graph
        dataPreprocDir = os.path.join(self.outDir, "preprocessing", "dataset_train_valid_test_split_v0.1") # Transformers dataset utput from preproc step
        flag = os.path.isdir(dataPreprocDir)
        if( flag ):
        	ds = load_from_disk(dataPreprocDir)

	        ds1 = self.gen_id('wfdataset') 
	        g.add( ( self.nerwf[ds1], self.RDF.type, self.xmlpo.LabeledDataset ) )
	        g.add( ( self.nerwf[ds1], self.RDFS.label, Literal( "Dataset splitted into train, test, and validation", lang="en") ) )
	        
	        # Features
	        fe = self.gen_id('feature') 
	        g.add( ( self.nerwf[fe], self.RDF.type, self.xmlpo.Feature ) )
	        g.add( ( self.nerwf[fe], self.RDFS.label, Literal( "Dataset feature ID", lang="en") ) )
	        datc = self.gen_id('featureCharacteristics') 
	        g.add( ( self.nerwf[datc], self.RDF.type, self.xmlpo.FeatureCharacteristic ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("id", lang="en") ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Original file identifier for the sentences", lang="en" ) ) )
	        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )

	        fe = self.gen_id('feature') 
	        g.add( ( self.nerwf[fe], self.RDF.type, self.xmlpo.Feature ) )
	        g.add( ( self.nerwf[fe], self.RDFS.label, Literal( "Dataset feature TOKENS", lang="en") ) )
	        datc = self.gen_id('featureCharacteristics') 
	        g.add( ( self.nerwf[datc], self.RDF.type, self.xmlpo.FeatureCharacteristic ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("id", lang="en") ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Original sentence tokens", lang="en" ) ) )
	        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )
	        
	        fe = self.gen_id('feature') 
	        g.add( ( self.nerwf[fe], self.RDF.type, self.xmlpo.Feature ) )
	        g.add( ( self.nerwf[fe], self.RDFS.label, Literal( "Dataset feature NER_TAGS", lang="en") ) )
	        datc = self.gen_id('featureCharacteristics') 
	        g.add( ( self.nerwf[datc], self.RDF.type, self.xmlpo.FeatureCharacteristic ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.FeatureName, Literal("id", lang="en") ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.FeatureDescription, Literal( "Correct entity labels for each token in the sentences", lang="en" ) ) )
	        g.add( ( self.nerwf[fe], self.nerwf.describedBy, self.nerwf[datc] ) )

	        # Data splits
	        dtrain = self.gen_id('wfdata')
            self.dtrain = dtrain
	        g.add( ( self.nerwf[dtrain], self.RDF.type, self.xmlpo.TrainSet ) )
	        g.add( ( self.nerwf[dtrain], self.RDFS.label, Literal( "Train data", lang="en") ) )
	        datc = self.gen_id('dataCharacteristics') 
	        g.add( ( self.nerwf[datc], self.RDF.type, self.xmlpo.DatasetCharacteristic ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.NumberOfFeatures, Literal(3) ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.NumberOfInstances, Literal( len( ds['train'] ) ) ) )
	        g.add( ( self.nerwf[dtrain], self.nerwf.describedBy, self.nerwf[datc] ) )

	        dtest = self.gen_id('wfdata') 
            self.dtest = dtest
	        g.add( ( self.nerwf[dtest], self.RDF.type, self.xmlpo.TestSet ) )
	        g.add( ( self.nerwf[dtest], self.RDFS.label, Literal( "Test data", lang="en") ) )
	        g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasData, self.nerwf[dtest] ) )
	        datc = self.gen_id('dataCharacteristics') 
	        g.add( ( self.nerwf[datc], self.RDF.type, self.xmlpo.DatasetCharacteristic ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.NumberOfFeatures, Literal(3) ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.NumberOfInstances, Literal( len( ds['test'] ) ) ) )
	        g.add( ( self.nerwf[dtest], self.nerwf.describedBy, self.nerwf[datc] ) )
	        
	        dvalid = self.gen_id('wfdata') 
	        g.add( ( self.nerwf[dvalid], self.RDF.type, self.nerwf.ValidationSet ) )
	        g.add( ( self.nerwf[dvalid], self.RDFS.label, Literal( "Validation data", lang="en") ) )
	        g.add( ( self.nerwf[ds1], self.xmlpo.datasetHasData, self.nerwf[dvalid] ) )
	        datc = self.gen_id('dataCharacteristics') 
	        g.add( ( self.nerwf[datc], self.RDF.type, self.xmlpo.DatasetCharacteristic ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.NumberOfFeatures, Literal(3) ) )
	        g.add( ( self.nerwf[datc], self.xmlpo.NumberOfInstances, Literal( len( ds['valid'] ) ) ) )
	        g.add( ( self.nerwf[dvalid], self.nerwf.describedBy, self.nerwf[datc] ) )
	        
        self.graph = g

        return flag, ds1

    def _describe_data_preprocessing(self):
        g = self.graph
        
        flag, dataset_id = self.__describe_preproc_dataset()
        if(flag):
	        proc1 = self.gen_id('wfoperation') # Data Handling => Parsing
	        g.add( ( self.nerwf[proc1], self.RDF.type, self.edam.operation_1812 ) )
	        g.add( ( self.nerwf[proc1], self.RDFS.label, Literal("Process texts and convert annotations to coNLL format", lang="en") ) )
	        g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )

	        proc2 = self.gen_id('wfoperation') # Data Handling => Formatting
	        g.add( ( self.nerwf[proc2], self.RDF.type, self.edam.operation_0335 ) )
	        g.add( ( self.nerwf[proc2], self.RDFS.label, Literal("Format coNLL annotations into IOB format", lang="en") ) )
	        g.add( ( self.nerwf[proc2], self.nerwf.applyTaggingFormat, self.nerwf.nlpformat_iob ) )
	        g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc2] ) )
			
            preprocParamConfig = self.gen_id('paramSetting') 
            g.add( ( self.nerwf[preprocParamConfig], self.RDF.type, self.xmlpo.ParameterSettings ) )
            g.add( ( self.nerwf[preprocParamConfig], self.RDFS.label, Literal("Proprocessing parameter configuration", lang="en") ) )
            g.add( ( self.nerwf[proc2], self.xmlpo.HasParameter, self.nerwf[preprocParamConfig] ) )
            
            value = True
            if('eliminate_overlappings' in self.config):
                if( self.config["eliminate_overlappings"] in [True, False] ):
                    value = self.config["eliminate_overlappings"]
            param = self.gen_id('preprocparam') 
            g.add( ( self.nerwf[param], self.RDF.type, self.xmlpo.ParameterCharacteristic ) )
            g.add( ( self.nerwf[param], self.xmlpo.ParameterName, Literal( "eliminate entity annotation overlapping", lang="en") ) )
            g.add( ( self.nerwf[param], self.xmlpo.ParameterValue, Literal(value) ) )
            g.add( ( self.nerwf[preprocParamConfig], self.xmlpo.hasParameterCharacteristic, self.nerwf[param] ) )

			proc3 = self.gen_id('wfoperation') # DatasetSplit
	        g.add( ( self.nerwf[proc3], self.RDF.type, self.xmlpo.DatasetSplit ) )
	        g.add( ( self.nerwf[proc3], self.RDFS.label, Literal("Reorganize the data according to the train, test and validation sets", lang="en") ) )
	        g.add( ( self.nerwf[proc3], self.nerwf.generatesDataset, self.nerwf[dataset_id] ) )
	        g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc3] ) )
			
        self.graph = g

    def __check_model_details(self, train_id):
        g = self.graph

        models = []
    	model_checkpoint = self.config["pretrained_model"]
    	hyperparams = {'per_device_train_batch_size': 8, 'per_device_eval_batch_size': 16, 'learning_rate': 2e-5, 'weight_decay': 0.01 }
		model_name = model_checkpoint.split("/")[-1]
		outDir = f"{outDir}/{model_name}-finetuned-ner"
		trout = os.path.join( outDir, "training" )
		flag = os.path.exists(trout)
		if(flag):
			model_files = [file for file in os.listdir(outDir) if file.startswith("model_")]
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

                trainParamConfig = self.gen_id('paramSetting') 
                g.add( ( self.nerwf[trainParamConfig], self.RDF.type, self.xmlpo.ParameterSettings ) )
                g.add( ( self.nerwf[trainParamConfig], self.RDFS.label, Literal("Training hyper parameter configuration", lang="en") ) )
                g.add( ( self.nerwf[train_id], self.xmlpo.HasParameter, self.nerwf[trainParamConfig] ) )
                for hp in hyperparams:
                    value = hyperparams[hp]
                    param = self.gen_id('hyperparam') 
                    g.add( ( self.nerwf[param], self.RDF.type, self.xmlpo.ParameterCharacteristic ) )
                    g.add( ( self.nerwf[param], self.xmlpo.ParameterName, Literal(hp, lang="en") ) )
                    g.add( ( self.nerwf[param], self.xmlpo.ParameterValue, Literal(value) ) )
                    g.add( ( self.nerwf[trainParamConfig], self.xmlpo.hasParameterCharacteristic, self.nerwf[param] ) )
                
                rootmodel = self.gen_id('original_model')
                g.add( ( self.nerwf[rootmodel], self.RDF.type, self.mesh.D000098342 ) )
                g.add( ( self.nerwf[rootmodel], self.RDFS.label, Literal( f"Original model {model_checkpoint}", lang="en") ) )
                for i, model in enumerate(model_files):
                    finemodel = self.gen_id('finetuned_model')
                    models.append(finemodel)
                    g.add( ( self.nerwf[finemodel], self.RDF.type, self.mesh.D000098342 ) )
                    g.add( ( self.nerwf[finemodel], self.RDFS.label, Literal( f"Finetunned {model}", lang="en") ) )
                    g.add( ( self.nerwf[finemodel], self.nerwf.finetunedFrom, self.nerwf[rootmodel] ) )
                    

        self.graph = g

		return flag, models	

    def _integrate_model_evaluation_agg_results(self, eval_op_id):
        task = 'ner'
        model_name = self.config['model_checkpoint'].split("/")[-1]
        fout = self.config['outpath']
        outDir = os.path.join(fout, f"{self.config['identifier']}-{model_name}-finetuned-{task}" )

        stage = 'test' # change to train later
        folder = os.path.join( outDir, stage, 'summary_reports' )
        
        g = self.graph

        eval_links = { 'accuracy': self.stato.0000415, 'precision': self.stato.0000416, 'recall': self.stato.0000233, 'aucroc': self.stato.0000608, 'f1-score': self.stato.0000628, 'mcc': self.stato.0000524, 'kappa': self.stato.0000630 }
        evaluation_modes = ['seqeval-default', 'seqeval-strict', 'sk-with-prefix', 'sk-without-prefix']
        levels = ['token', 'word']
        for mode in evaluation_modes:
            for level in levels:
                files = glob.glob( os.path.join(folder, f"{mode}_summary-report_*-l{level}.tsv") )
                if( len(files) > 0 ):
                    datc = self.gen_id('evalCharacteristics') 
                    g.add( ( self.nerwf[datc], self.RDF.type, self.xmlpo.ModelEvaluationCharacteristic ) )
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

                            score = self.gen_id('evalScore') 
                            g.add( ( self.nerwf[score], self.RDF.type, self.nerwf.NEREvaluationMeasure ) )
                            g.add( ( self.nerwf[score], self.nerwf.belongsToEntity, Literal(entity) ) )
                            g.add( ( self.nerwf[score], self.nerwf.isAggregatedValue, Literal(True) ) )
                            g.add( ( self.nerwf[score], self.vcard.hasValue, Literal( value ) ) )
                            g.add( ( self.nerwf[score], self.nerwf.aggregatedByStatsFunction, Literal( statsMetric, lang="en" ) ) )
                            if( evalMetric in eval_links ):
                                g.add( ( self.nerwf[score], self.nerwf.fromEvaluationMetric, eval_links[evalMetric] ) )
                            g.add( ( self.nerwf[datc], self.nerwf.hasScore, self.nerwf[score] ) )

        self.graph = g
    
    def _describe_training(self):
        g = self.graph

        proc1 = self.gen_id('wfoperation')
        flag, models = self.__check_model_details(proc1)
        if(flag):
	        g.add( ( self.nerwf[proc1], self.RDF.type, self.xmlpo.Train ) )
	        g.add( ( self.nerwf[proc1], self.RDFS.label, Literal("Training step", lang="en") ) )
	        g.add( ( self.nerwf[proc1], self.nerwf.useInputData, self.nerwf[self.dtrain] ) )
            
            approach = self.gen_id('mlapproach')
            g.add( ( self.nerwf[approach], self.RDF.type, self.xmlpo.ClassificationAlgorithm ) )
            g.add( ( self.nerwf[approach], self.RDFS.label, Literal("Deep Learning - Transformers", lang="en") ) )
            g.add( ( self.nerwf[proc1], self.stato.0000102, self.nerwf[approach] ) ) # operation executes 
            
            for model_id in models:
                g.add( ( self.nerwf[approach], self.xmlpo.hasOutput, self.nerwf[model_id] ) )
                g.add( ( self.nerwf[proc1], self.nerwf.generatesModel, self.nerwf[model_id] ) )
	        g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )

            proc1 = self.gen_id('wfoperation')
            g.add( ( self.nerwf[proc1], self.RDF.type, self.xmlpo.MLModelEvaluation ) )
            g.add( ( self.nerwf[proc1], self.RDFS.label, Literal("Training evaluation", lang="en") ) )
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )
            self._integrate_model_evaluation_agg_results(proc1)

        self.graph = g

    def _describe_test(self):
        g = self.graph

        flag = self.__check_results_test_details()
        if(flag):
	        proc1 = self.gen_id('wfoperation') 
	        g.add( ( self.nerwf[proc1], self.RDF.type, self.xmlpo.Test ) )
	        g.add( ( self.nerwf[proc1], self.RDFS.label, Literal("Test step", lang="en") ) )
	        g.add( ( self.nerwf[proc1], self.nerwf.useInputData, self.nerwf[self.dtest] ) )
            g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )

        self.graph = g

    def _describe_prediction(self):
        g = self.graph

        flag = self.__check_results_prediction_details()
        if(flag):
	        proc1 = self.gen_id('wfoperation') # edam - Prediction and recognition
	        g.add( ( self.nerwf[proc1], self.RDF.type, self.edam.operation_2423 ) )
	        g.add( ( self.nerwf[proc1], self.RDFS.label, Literal("Prediction step on new data", lang="en") ) )
	        g.add( ( self.nerwf[self.wfid], self.nerwf.containsProcedure, self.nerwf[proc1] ) )


        self.graph = g


    def run(self):
    	self._describe_experiment()
    	self._describe_data_preprocessing()
    	self._describe_training()
    	self._describe_test()
    	self._describe_prediction()

        # describe experiment
        # describe workflow
        # describe preprocess
        # describe train
        # describe test
        # describe prediction

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
                    useInputData instanceTrainSet (DatasetOutTrain)
                    generatesScoreDataset score_dataset
                    self:hasScore self:NEREvaluationMeasure (nscore1)
                
                nscore1 self:belongsToEntity "intervention"
                nscore1 fromEvaluationMetric f1, acc, mcc
                nscore1 isAggregatedValue true
                nscore1 aggregatedByStatsFunction max (stato:0000151), min(stato:0000150), mean (ncit:C53319), std (ncit:C53322), median (ncit:c28007)
                nscore1 vcard:hasValue 0.8

                ModelEvaluationTechnique describedBy ModelEvaluationCharacteristic
                modelEvaluation1 a ModelEvaluationCharacteristic
                modelEvaluation1 xmlpo:MLEvaluationTechniqueName sklearn_without-prefix
                modelEvaluation1 xmlpo:MLEvaluationTechniqueDescription sk learn without the prefixes

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