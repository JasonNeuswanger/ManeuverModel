import os
import re
from sys import argv

# CONFIGURATION FOR A GIVEN JOB

N_NODES = int(argv[1])
CORES_PER_NODE = 1 # Note the docs say this should be a multiple of 3, although some examples use ppn=1
JOB_NAME = argv[2]

# Example usage:
# python maneuver_batch_builder.py 5 'InitialTry'

# COMMON CODE FOR ANY JOB

for i in range(N_NODES):
    batch_name_raw = "{0} Node {1}".format(JOB_NAME, i)
    batch_name = re.sub(r'\W+', '', batch_name_raw)
    # formerly had this one, can't find a translation: #PBS -S /bin/bash
    # formerly PBS -l nodes=1:ppn={cores}:Intel
    # partition should be the name of the queue, which is batch?
    # previously requested 500mb memory
    batch_file_contents = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=168:00:00
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jasonneuswanger@gmail.com

echo
echo "Job ID: $SLURM_JOB_ID"
echo "Queue (Partition):  $SLURM_JOB_PARTITION"
echo "Cores:  $SLURM_NTASKS"
echo "Nodes:  $($SLURM_JOB_NODELIST | sort -u | tr '\n' ' ')"
echo

python3 maneuver_spreadsheet_creation.py
    """.format(cores=CORES_PER_NODE, job_name=JOB_NAME)

    with open('{0}.sh'.format('batches/' + batch_name), 'w') as batch_file:
        batch_file.write(batch_file_contents)

    os.system("sbatch 'batches/{0}.sh'".format(batch_name)) # no comment