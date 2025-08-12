import os
import sys
from autologging import logged

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )
from utils.config import Config
from utils.hpc import HPC
