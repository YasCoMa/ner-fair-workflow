nextflow.enable.dsl=2

params.mode="all"
params.dataDir="/mnt/085dd464-7946-4395-acfd-e22026d52e9d/home/yasmmin/Dropbox/irbBCN_job/match_clinical_trial/experiments"

params.help = false
if (params.help) {
    log.info params.help_message
    exit 0
}

log.info """\
 NER Fair workflow  -  P I P E L I N E
 ===================================
 dataDir       : ${params.dataDir}
 runningConfig : ${params.runningConfig}
 mode       : ${params.mode}
 """

process setEnvironment {
    
    output:
    val 1, emit: flag

    script:
        """
        if [ ! -d "${params.dataDir}" ]; then
            mkdir ${params.dataDir}
        fi
        """
}

process PROCESS_PreProcessTasks {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_data_preprocessing.ready"
        
    script:
        "python3 ${projectDir}/modules/preprocessing/preprocessing.py -execDir $outDir -paramFile $parameterFile "
}

process PROCESS_TrainTasks {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_training.ready"
        
    script:
        "python3 ${projectDir}/modules/evaluation/training.py -execDir $outDir -paramFile $parameterFile "
}

process PROCESS_TestTasks {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_test.ready"
        
    script:
        "python3 ${projectDir}/modules/evaluation/test.py -execDir $outDir -paramFile $parameterFile "
}

process PROCESS_PredictionTasks {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_prediction.ready"
        
    script:
        "python3 ${projectDir}/modules/evaluation/prediction.py -execDir $outDir -paramFile $parameterFile "
}

process PROCESS_MetaEnrichmentTasks {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_semantic_description.ready"
        
    script:
        "python3 ${projectDir}/modules/semantic_description.py -execDir $outDir -paramFile $parameterFile "
}

process PROCESS_SignificanceAnalysisTasks {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_significance_analysis.ready"
        
    script:
        "python3 ${projectDir}/modules/significance_analysis.py -execDir $outDir -paramFile $parameterFile "
}

workflow {
    result = setEnvironment()

    if( params.mode == "preprocess" | params.mode == "all" ){
        result = PROCESS_PreProcessTasks( params.dataDir, params.runningConfig, result )
    }

    if( params.mode == "training" | params.mode == "all" ){
        result = PROCESS_TrainTasks( params.dataDir, params.runningConfig, result )
    }

    if( params.mode == "test" | params.mode == "all" ){
        result = PROCESS_TestTasks( params.dataDir, params.runningConfig, result )
    }

    if( params.mode == "prediction" | params.mode == "all" ){
        result = PROCESS_PredictionTasks( params.dataDir, params.runningConfig, result )
    }

    if( params.mode == "metadata_enrichment" | params.mode == "all" ){
        result = PROCESS_MetaEnrichmentTasks( params.dataDir, params.runningConfig, result )
    }

    if( params.mode == "significance_analysis" ){
        result = PROCESS_SignificanceAnalysisTasks( params.dataDir, params.runningConfig, result )
    }

}

// nextflow run -bg /aloy/home/ymartins/match_clinical_trial/ner_subproj/main.nf --dataDir /aloy/home/ymartins/match_clinical_trial/experiments/longformer/ --runningConfig /aloy/home/ymartins/match_clinical_trial/experiments/config_biobert.json --mode "all"

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Models were trained and evaluated!\n" : "Oops .. something went wrong" )
}
