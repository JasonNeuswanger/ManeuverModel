import os
import re
from sys import argv

# CONFIGURATION FOR A GIVEN JOB

N_NODES = int(argv[1])
CORES_PER_NODE = 1
JOB_NAME = argv[2]

# Example usage:
# python maneuver_batch_builder.py 5 'InitialTry'

# COMMON CODE FOR ANY JOB

for i in range(N_NODES):
    batch_name_raw = "{0} Node {1}".format(JOB_NAME, i)
    batch_name = re.sub(r'\W+', '', batch_name_raw)
    batch_file_contents = """
    #PBS -S /bin/bash
    #PBS -N {job_name}
    #PBS -q batch
    #PBS -l nodes=1:ppn={cores}:Intel
    #PBS -l walltime=96:00:00
    #PBS -l mem=8gb
    #PBS -M jasonneuswanger@gmail.com
    #PBS -m ae

    cd $PBS_O_WORKDIR

    echo
    echo "Job ID: $PBS_JOBID"
    echo "Queue:  $PBS_QUEUE"
    echo "Cores:  $PBS_NP"
    echo "Nodes:  $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
    echo

    echo "Python path is: "
    printenv PYTHONPATH
    
    echo "Python version is"
    python3 --version

    python3 maneuver_spreadsheet_creation.py
    """.format(cores=CORES_PER_NODE, job_name=JOB_NAME)

    with open('{0}.sh'.format('batches/' + batch_name), 'w') as batch_file:
        batch_file.write(batch_file_contents)

    os.system("qsub 'batches/{0}.sh'".format(batch_name)) # no comment