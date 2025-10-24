# NER Fair wf - Workflow to perform end-to-end named entity recognition experiments

Nextflow workflow to perform named entity recognition experiments handling since the dataset preparation step, the training by finetunning large language models, test, prediction and metadata enrichment describing all available about both the experiment itself and data generated/processed along the pipeline. It also allows for the models comparison across related experiments through p-value computation.

<div style="text-align: center">
	<img src="pipeline.png" alt="pipeline"
	title="NER Fair workflow" width="550px" />
</div>

## Summary

We have developed a comprehensive workflow with six functionalities: (1) Data transformation from a set of files paired as docid.ann (entity annotation) and docid.txt (respective plain raw text); (2) Training with possibility of choosing the llm model to finetune, option of reusing a json hyperparameter file, perform hyperparameter search or just using pre-defined hyperparameter values; (3) Test using the test split part that was processed in (1) or test using a custom external dataset (transformers compatible); (4) Prediction from a folder of plain raw txt files, producing a consensus list of annotations containing only those that model replicates agreed with a minimum mean score of 0.8; (5) Experiment metadata exportation, in which the experiment metadata provided in the configuration file plus the information processed/generated along the workflow execution will be described with our proposed ontology elements for NLP experiments; (6) Analysis of significance across scope-related experiments, supporting the stratification per entity or calculating the global significance per evaluation metric (f1-score, accuracy, precision, etc.), according to the following illustration.
<div style="text-align: center">
	<img src="external_evaluators_explanation.png" alt="significance"
	title="Significance analysis options" width="200px" />
</div>



## Requirements:
* The packages are stated in the environment's exported file: environment.yml

## Usage Instructions
### Preparation:
1. ````git clone https://github.com/YasCoMa/ner-fair-workflow.git````
2. ````cd ner-fair-workflow````
3. ````conda env create --file environment.yml````
4. The workflow requires four parameters, you can edit them in main.nf, or pass in the command line when you start the execution. The parameters are:
	- **mode**: Indicates the goal of the workflow: 'preprocess', 'training', 'test', 'prediction', 'metadata_enrichment' or 'significance_analysis'. It activates accordingly the steps according to the mode.
	- **dataDir**: The directory where the generated files will be stored
	- **runningConfig**: A json file with the configuration setup desired by the user. Each main key of the json file is explained below.
		- datasets: The datasets available for analysis - Example: ``["hla", "bcipep", "gram+_epitope", "gram-_epitope", "gram+_protein", "gram-_protein", "allgram_epitope", "allgram_protein"]``


### Run workflow:
1. Examples of running configuration are shown in running_config.json and eskape_running_config.json

2. Modes of execution:
	- **Run All:**
		- ````nextflow run main.nf```` or ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode all ````
	- **Run Pre-processing:**
		- ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode preprocess ````
	- **Run Training:**
		- ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode training````
	- **Run Test:**
		- ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode test````
	- **Run Prediction:**
		- ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode prediction````
	- **Run Metadata Enrichment:**
		- ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode metadata_enrichment````
	- **Run Significance analysis:**
		- ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode significance_analysis````

## Reference

## Bug Report
Please, use the [Issues](https://github.com/YasCoMa/ner-fair-workflow/issues) tab to report any bug.