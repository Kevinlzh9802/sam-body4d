#!/bin/bash
#SBATCH --partition=insy,general # Request partition. Default is 'general' 
#SBATCH --qos=medium         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=34:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --mem=8G
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=/home/nfs/zli33/slurm_outputs/sam_3d_body/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/nfs/zli33/slurm_outputs/sam_3d_body/slurm_%j.err # Set name of error log. %j is the Slurm jobId

#SBATCH --gres=gpu:a40:1 # Request 1 GPU
mnt_path=/tudelft.net/staff-umbrella/neon/
bind_mnt_path=/mnt
sif_path=$bind_mnt_path/apptainer/sam-body4d.sif

project_folder=$bind_mnt_path/zonghuan/projects/sam-body4d
input_folder=$bind_mnt_path/zonghuan/data/sam4d_body/inputs
output_folder=$bind_mnt_path/zonghuan/data/sam4d_body/outputs
kp_folder=$bind_mnt_path/datasets/sam_3d_body/bboxes_kps

apptainer exec --nv --bind $mnt_path:$bind_mnt_path $sif_path python $sam_3d_body_path/demo.py --image_folder $input_folder --output_folder $output_folder 