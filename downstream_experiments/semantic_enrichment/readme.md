# Pipeline for metadata enrichment and datasets knowledge graph fusion

This is an auxiliary pipeline to execute the metadata enrichment module in the NER Fair workflow for each dataset that was used from the benchmark (bc5cdr, ncbi, biored, chia_without_scope, chia_without_scope), and an extra dataset corresponding to the merged version of these datasets. Later, it merges all the semantic graphs into onw knowledge graph.

## Execution
The package dependencies are covered by the conda environment configuration file in the root of this repository (environment.yml).

This analysis assumes that the benchmark dataset experiments were already executed.

Running analysis: `python3 merge_experiment_graphs.py _outFolder_     _expRootFolder_` , where \_outFolder\_ is the path to a custom output folder. If this parameter is empty, it will save in the default path "./merged_experiment_graphs". The \_expRootFolder\_ if the path to the output folder used in the pipeline found in "prepare_benchmark_datasets" folder. If this parameter is empty, it will assume the default path "./preprocessed_datasets_for_workflow"

## Outputs explanation
In the example "merged_experiment_graphs" folder, you will find the following output items:
- all_nerfair_graph.ttl : the merged knowledge graph exported in turtle

- all_nerfair_graph.xml : the merged knowledge graph exported in rdf/xml