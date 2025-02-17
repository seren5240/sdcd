#!/bin/bash
#
#SBATCH --account=pi-naragam
#SBATCH --job-name=sdcd_benchmark
#SBATCH --output=./slurm/out/%j.%N.stdout
#SBATCH --error=./slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/skwok1/sdcd/tutorials
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000
#SBATCH --time=3-00:00:00

set -e

module load python/booth/3.10

cd ..
pip3 install -e .
pip3 install igraph
cd tutorials

echo "Running benchmark"

# python3 -u "./benchmark.py" 10 1000 1
# python3 -u "./benchmark.py" 10 1000 2
python3 -u "./benchmark.py" 10 1000 4
