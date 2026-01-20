#!/bin/bash


#SBATCH --job-name=adding_2_and_2
#SBATCH --output=addition.log
#SBATCH --error=addition.err
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=all
#SBATCH --nodelist=ceta1
#SBATCH --mem=80G





python3 add_2_to_2.py > my_result.txt \

