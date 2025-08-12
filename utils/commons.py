import os
import sys
import glob
import shutil
import subprocess
from subprocess import call, Popen

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )
from utils import HPC, Config

def readParametersFile( aParamsFilename ):
    params = {}
    paramsFile = open( aParamsFilename )
    for line in paramsFile:
        if len(line.strip()) == 0: continue
        if line.strip()[0] == "#": continue
        name,value = line.strip().split()[:2]
        params[name] = value
    paramsFile.close()

    os.environ["UPDATE_CONFIG"] = params["UPDATE_CONFIG"]

    return params

def execAndCheck(cmdStr, allowedReturnValues=set()):
    try:
        retcode = call(cmdStr,shell=True)
        if retcode != 0:
            if retcode not in allowedReturnValues:
                if retcode > 0:
                    sys.stderr.write( "ERROR return code %d, please check!\n" % retcode )
                elif retcode < 0:
                    sys.stderr.write( "Command terminated by signal %d\n" % -retcode )
                sys.exit(1)
    except OSError as e:
        sys.stderr.write( "Execution failed: %s\n" % e )
        sys.exit(1)
    return retcode
    
def check_previous_job( job_path, current ):
    flag = True 

    if(job_path.endswith('/')):
        job_path = job_path[:-1]
    root = '/'.join( job_path.split('/')[:-1] )

    parts = current.split('_')
    step = parts[0].replace('J','').replace('j','').split('.')
    jid = str( int( step[-1] ) - 1 )
    if( len(jid)==1 ):
        jid = '0'+jid

    init = '.'.join( [ step[0], step[1], jid ] )
    prev = init
    prevfolder = list( filter( lambda x: x.startswith(init), os.listdir(root) ))
    flag = ( len(prevfolder) > 0 or jid == '-1' )

    if( flag and jid != '-1' ):
        prev = prevfolder[0]
        prevfolder = os.path.join( root, prev )
        prevmarker = os.path.join( prevfolder, prev+'.ready' )
        if( not os.path.isfile(prevmarker) ):
            flag = False

    if( not flag ):
        sys.stderr.write( "Previous step job was not completed: %s\n" % prev )
        sys.exit(1)
    return flag

def monitor_background_job( job_path, job_name ):
    job_id = 0
    sge = False
    log_files = list( filter( lambda x: x.startswith('slurm-'), os.listdir( job_path ) ))
    cnt = len( log_files )
    if( cnt == 0 ):
        log_files = list( filter( lambda x: x.find( job_name+'.o') != -1, os.listdir( job_path ) ))
        cnt = len( log_files )
        if( cnt > 0 ):
            sge = True
            job_id = log_files[0].split('.')[-2][1:]
    else:
        job_id = log_files[0].split('.')[0].split('-')[1].split('_')[0]

    if(job_id != 0 ):
        config = Config()
        cluster = HPC.from_config( config )
        cluster.monitor_background_job( job_path, job_name, job_id)

def prepare_launch_monitor_job(job_path, job_name):
    command =  """
import os
from utils.commons import *

job_path = '"""+job_path+"""'
job_name = '"""+job_name+"""'

started = os.path.join( job_path, job_name+'.started' )
ready = os.path.join( job_path, job_name+'.ready' )
error = os.path.join( job_path, job_name+'.error' )
while( os.path.isfile(started) and ( not os.path.isfile(ready) ) and ( not os.path.isfile(error) ) ):
    monitor_background_job( job_path, job_name )
"""
    script = os.path.join(job_path, job_name+'_monitor_job.py')
    f = open( script, 'w' )
    f.write(command)
    f.close()

    subprocess.Popen(["python", script])

def prepare_job_single( job_name, job_path, command, ignore_check = False, wait = True, destroy=True, ncpus=1, nmem=20, execpy='python', hpc_env='slurm', config_path=None ):
    
    flag = True
    if(not ignore_check):
        flag = check_previous_job( job_path, job_name)

    if( (config_path is None) or ( not os.path.exists(config_path) ) ):
        config_path = os.environ['HPC_CONFIG']
        '''
        if(hpc_env=='slurm'):
            config = Config('/aloy/home/ymartins/updatedbs/adapt_i3d_3did_irbcluster/config.json') # os.environ['UPDATE_CONFIG']
        else:
            config = Config('/aloy/home/ymartins/updatedbs/adapt_i3d_3did_irbcluster/config_update_sge.json')
        '''
    config = Config(config_path)

    if(os.path.isdir(job_path) and flag ):
        ready = os.path.join( job_path, job_name+'.ready' )
        error = os.path.join( job_path, job_name+'.error' )
        if(os.path.isfile(ready)):
            flag = False
        else:
            if(destroy):
                shutil.rmtree(job_path)
    if(flag):
        template_script = None
        if( len(command.split('\n')) > 1 ):
            template_script = command
            command = None

        elements = []
        n_jobs = 1
        #config = Config()
        cluster = HPC.from_config( config )
        cluster.create_submit_job( job_name, n_jobs, job_path=job_path, elements = elements, template_script = template_script, command = command, config=config, wait=wait, ncpus=ncpus, nmem=nmem, execpy=execpy)

def prepare_job_array( job_name, job_path, command, filetasksFolder="", taskList=[], chunk_size=10, ignore_check = False, hpc_env='slurm', wait=True, destroy=True, ncpus=1, nmem=10, execpy='python', config_path=None ):
    flag = True
    if(not ignore_check):
        flag = check_previous_job( job_path, job_name)

    if( (config_path is None) or ( not os.path.exists(config_path) ) ):
        config_path = os.environ['HPC_CONFIG']
        '''
        if(hpc_env=='slurm'):
            config = Config('/aloy/home/ymartins/updatedbs/adapt_i3d_3did_irbcluster/config.json') # os.environ['UPDATE_CONFIG']
        else:
            config = Config('/aloy/home/ymartins/updatedbs/adapt_i3d_3did_irbcluster/config_update_sge.json')
        '''
    config = Config(config_path)

    if(os.path.isdir(job_path) and flag ):
        ready = os.path.join( job_path, job_name+'.ready' )
        error = os.path.join( job_path, job_name+'.error' )
        if(os.path.isfile(ready)):
            flag = False
        else:
            if(destroy):
                shutil.rmtree(job_path)
    if(flag):
        template_script = None
        if( len(command.split('\n')) > 1 ):
            template_script = command
            command = None

        total_tasks = 0
        elements = taskList
        if( len(taskList) == 0 and (filetasksFolder is not None) and (filetasksFolder != "") ):
            if( os.path.isdir(filetasksFolder) ):
                filetasksFolder = os.path.join(filetasksFolder, 'tasks')
                for f in os.listdir(filetasksFolder):
                    total_tasks += 1
                    elements.append( os.path.join(filetasksFolder, f) )

            elif( os.path.isfile(filetasksFolder) ):
                g = open(filetasksFolder, 'r')
                for line in g:
                    if(line != ''):
                        total_tasks += 1
                        line = line.replace('\n','')
                        elements.append( line )
                g.close()
                
        total_tasks = len(elements)

        ssplit = len(elements) #int( round( len(elements) * 0.5 ) )
        add_cnf = ''
        if( hpc_env == 'sge' ):
            '''
            chunk_size = 5
            elements = elements[ssplit:]
            '''
            n_jobs = max( total_tasks // chunk_size, 1)
            user = config.HPC.username.replace("'", "")
            add_cnf = '-H /aloy/home/{} --bind /aloy/home,/aloy/web_repository,/aloy/scratch,/aloy/data,/apps '.format(user)

        if( hpc_env == 'slurm' ):
            '''
            chunk_size = chunk_size
            elements = elements[:ssplit]
            '''

            n_jobs = max( total_tasks // chunk_size, 1)
            while( n_jobs >= 5000 ):
                chunk_size = chunk_size*2
                n_jobs = total_tasks // chunk_size

        cluster = HPC.from_config( config )
        cluster.create_submit_job( job_name, n_jobs, force_array = True, job_path=job_path, elements = elements, template_script = template_script, command = command, config = config, add_cnf=add_cnf, wait=wait, ncpus=ncpus, nmem=nmem, execpy=execpy )
        if(wait == False):
            prepare_launch_monitor_job( job_path, job_name)
        
def compressJobFiles(directory,jobName,additionalFileSpecs):
    currentPath = os.getcwd()
    os.chdir(directory)

    cmdStr = 'tar czf '+jobName+'.tgz '
    flag = True 
    for fs in additionalFileSpecs:
        flag = flag and ( len( glob.glob(fs) ) > 0 )

    if( flag ):
        for fs in additionalFileSpecs: cmdStr += ' '+fs
        execAndCheck(cmdStr)

        cmdStr = 'rm -rf '
        for fs in additionalFileSpecs: cmdStr += ' '+fs
        execAndCheck(cmdStr)

    os.chdir(currentPath)
