# Pipeline for benchmark datasets preparation for the execution in the NER Fair workflow

This is an auxiliary pipeline to generate the input files required by the NER Fair workflow for each dataset that was used from the benchmark (bc5cdr, ncbi, biored, chia_without_scope, chia_without_scope), and an extra dataset corresponding to the merged version of these datasets.

## Execution
The package dependencies are covered by the conda environment configuration file in the root of this repository (environment.yml).

Running analysis: `python3 preprocess_datasets_benchmark.py _outFolder_` , where \_outFolder\_ is the path to a custom output folder. If this parameter is empty, it will save in the default path "./preprocessed_datasets_for_workflow"

## Outputs explanation
In the example "preprocessed_datasets_for_workflow" folder, you will find the following output items:
- "processed" folder : in each dataset subfolder you will find the .ann and .txt files accounting for the annotations and plain text files, respectively.

- "tags" folder : it contains a structured file for each dataset with the observed tags (entities) observed on the annotations of these datasets.

- "configs" folder : it contains a structured file for each dataset with the running configuration to correctly execute the experiment.

- "trials" folder : this folder is created empty to serve as the working directory for the workflow. It will be the root path that will contain the data folder of each experiment.

- commands.json : this structured file contains an array with the command lines ready to be executed to call the NER Fair workflow for each dataset.