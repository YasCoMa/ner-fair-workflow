import os
import glob
import json
import shutil
from tqdm import tqdm

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )

class PreprocessBenchmarkDatasets:
    def __init__(self, fout):
        self.fout =  fout

        self.dss =  [ "bc5cdr", "ncbi", "biored", "chia_without_scope", "chia_without_scope" ]
        subfolders = ["tags", "processed", "configs", "trials"]
        for sf in subfolders:
            os.makedirs( os.path.join(self.fout, sf) )

        self.raw_path = os.path.join( self.fout, "raw_data" ) 
        if( not os.path.exists(self.raw_path) ):
            raw = os.path.join(root_path, "downstream_experiments", "prepare_benchmark_datasets", "raw_data.tar.xz")
            os.system(f"tar xzf {raw} -C {self.fout}")

        # Uncompress the json files regarding the NCBI clinical trials raw data
        self.hyperparams = os.path.join( fout, "best_params.pkl" ) 
        if( not os.path.exists(self.hyperparams) ):
            chyperPath = os.path.join(root_path, "downstream_experiments", "prepare_benchmark_datasets", "best_params.pkl")
            shutil.copy(chyperPath, self.hyperparams)

    def _parse_pubtator(self, file, tags, tmp):
        #tmp = {}
        _id = ""
        txt = ""
        f = open(file, 'r')
        for l in f:
            l = l.replace('\n','')
            if( l.find('|t|') != -1 ):
                _id = l.split("|t|")[0]
                txt = l.split("|t|")[1]
                tmp[_id] = { 'txt': txt, "ann": [] }
            elif( l.find('|a|') != -1 ):
                _id = l.split("|a|")[0]
                tmp[_id]['txt'] += " " + l.split("|a|")[1]
            else:
                #ps = list( filter( lambda x: x!="", l.split(" ") ))
                ps = l.split('\t')
                if( len(ps) == 6 ):
                    _id, start, end, word, entity, instance_entity = ps
                    tags.add(entity)
                    '''
                    start = ps[1]
                    end = ps[2]
                    entity = ps[-2]
                    word = ' '.join( ps[3:-2] )
                    '''
                    n = len(tmp[_id]['ann']) + 1
                    tmp[_id]['ann'].append( f'T{n}\t{entity} {start} {end}\t{word}' )
        f.close()
    
        return tmp, tags
        
    def _parse_anntxt(self, file, tags, tmp):
        #tmp = {}
        init = file.replace(".ann", "")
        _id = init.split('/')[-1]
        txt = open( f'{init}.txt', 'r' ).read()
        ann = []
        f = open(file, 'r')
        for line in f:
            l = line.replace('\n','')
            if( l.startswith('T') ):
                uid, annot, word = l.split('\t')
                entity = annot.split(' ')[0]
                tags.add(entity)
                annot = ' '.join( annot.split(' ')[1:] )
                pos = annot.split(';')
                for p in pos:
                    n = len(ann) + 1
                    start, end = p.split(' ')
                    nl = f'T{n}\t{entity} {start} {end}\t{word}'
                    ann.append(nl)
        f.close()
        
        tmp[_id] = { 'txt': txt, "ann": ann }
        
        return tmp, tags
        
    def _write_output(self, fout, dat):
        for _id in dat:
            for ftype in dat[_id]:
                content = dat[_id][ftype]
                if( isinstance(content, list) ):
                    content = '\n'.join(content)
                opath = os.path.join(fout, f'{_id}.{ftype}' )
                f = open(opath, 'w')
                f.write(content)
                f.close()
                
    def _prepare_tags(self, dsr, tags):
        tout = os.path.join( self.fout, 'tags')
        opath = os.path.join(tout, f'tags_{dsr}.json')
        final_tags = ['O']
        for t in tags:
            final_tags.append(f'B-{t}')
            final_tags.append(f'I-{t}')
        
        json.dump(final_tags, open(opath, 'w') )
    
    def parse_datasets(self):
        tout = os.path.join( self.fout, 'tags')
            
        possible_formats = { '*.pubtator.txt': '_parse_pubtator', '*.ann': '_parse_anntxt' }
        
        dspath = self.raw_path
        dss = self.dss
        for ds in dss:
            dsr = ds.split("/")[0]
            fout = os.path.join(self.fout, 'processed', dsr)
            if( not os.path.isdir(fout) ):
                os.makedirs(fout)
                
            dat = {}
            tags = set()
            for formatt in possible_formats:
                func = possible_formats[formatt]
                path = os.path.join( dspath, ds, formatt )
                fs = glob.glob( path )
                if( len(fs) > 0 ):
                    for f in tqdm(fs):
                        dat, tags = eval( f"self.{func}( f, tags, dat )" )
                        
            self._write_output( fout, dat)
            self._prepare_tags(dsr, tags)

    def make_merged_dataset(self):
        dspath = self.raw_path
        dss = self.dss
        possible_formats = { '*.pubtator.txt': '_parse_pubtator', '*.ann': '_parse_anntxt' }
        modes = { 'train': { '*.pubtator.txt': ['train', 'dev'], '*.ann': 0.6 }, 'test': { '*.pubtator.txt': ['test'], '*.ann': -0.4 } }

        tags = set()
        for m in modes:
            fname = f'merged_{m}'
            fout = os.path.join( self.fout, fname)
            if( not os.path.isdir(fout) ):
                os.makedirs(fout)
            dat = {}
            for ds in dss:
                dsr = ds.replace("/", '_')
                for formatt in possible_formats:
                    func = possible_formats[formatt]

                    path = os.path.join( dspath, ds, formatt )
                    fs = glob.glob( path )
                    cutrule = modes[m][formatt]
                    if( isinstance(cutrule, list) ):
                        fs = list( filter( lambda x: x.split('/')[-1].split('.')[0] in cutrule, fs))
                    else:
                        ln = len(fs)
                        n = round( abs(cutrule) * ln )
                        if(cutrule < 0):
                            offset = round( (1 + cutrule) * ln )
                            fs = fs[offset:]
                        else:
                            fs = fs[:n]

                    if( len(fs) > 0 ):
                        for f in tqdm(fs):
                            dat, tags = eval( f"self.{func}( f, tags, dat )" )
            self._write_output( fout, dat)
        self._prepare_tags('merged', tags)

    def make_config(self):
        exec_path = os.path.join( self.fout, 'trials')
        tout = os.path.join( self.fout, 'tags')
        fout = os.path.join( self.fout, 'configs')
        hpath = self.hyperparams
        cmdpath = 
            
        commands = {}
        dss = [ "bc5cdr", "ncbi", "biored", "chia_without_scope", "chia_without_scope", "merged_train", "merged_test" ]
        for ds in dss:
            dsr = ds.replace("/", '_')
            opath = os.path.join(fout, f"config_{dsr}.json")
            tpath = os.path.join(tout, f'tags_{dsr}.json')
            
            if( not os.path.isfile(opath) or True ):
                dsfout = os.path.join( self.fout, 'processed', dsr)
                commands[dsr] = f"nextflow run -bg {root_path}/main.nf --dataDir {exec_path} --runningConfig {opath} --mode 'all' "
                
                tmp = { "identifier": f"biobert-{dsr}",
                    "outpath": rootpath,
			                    "pretrained_model": "dmis-lab/biobert-base-cased-v1.2",
			                    "target_tags": tpath,
			                    "data_text_path": dsfout,
			                    "eliminate_overlappings": True,
                                "seed": 42,
			                    "do_hyperparameter_search": False,
			                    "hyperparameter_path": hpath,
			                    "optimization_metric": "f1",
                    "report_summary_stats_metric": "median",
                    "experiment_metadata": { 
                        "name": f"NER benchmark for  dataset {dsr}", 
                        "domain": "NER for PICO entities in texts concerning clinical trials"
                        }
                    }
                json.dump(tmp, open(opath, 'w') )

        json.dump(commands, open(cmdpath, 'w') )
    
    def run(self):
        self.parse_datasets()
        self.make_merged_dataset()
        self.make_config()

if( __name__ == "__main__" ):
    fout = "./preprocessed_datasets_for_workflow"
    if( len(sys.argv) > 1 ):
        fout = sys.argv[1]
    i = PreprocessBenchmarkDatasets( fout )
    i.run()
