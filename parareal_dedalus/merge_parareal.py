"""
Merge distributed analysis sets from a FileHandler.

Usage:
    merge.py <base_path> [--cleanup]

Options:
    --cleanup   Delete distributed files after merging

"""

import h5py
import h5py
import subprocess
from dedalus.tools import post
import pathlib
import sys
import os
import argparse
import glob
from docopt import docopt

args=docopt(__doc__)
base=args['<base_path>']
iter_paths=os.listdir(args['<base_path>'])


for path in iter_paths:
    print(base+"/"+path)
    folder=base+"/"+path
    post.merge_analysis(folder)
    


for path in iter_paths:
    print(base+"/"+path)
    set_paths=list(glob.glob(base+"/"+path+"/*h5"))
    print(set_paths)
    post.merge_sets(base+"/"+path+".h5",set_paths,cleanup=False)
    #~ print(file_name)


