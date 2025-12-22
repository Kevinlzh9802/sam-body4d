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
# local_path=/home/nfs/zli33
bind_mnt_path=/mnt/zonghuan
# bind_local_path=/mnt/zli33
sif_path=$bind_mnt_path/apptainers/detectron_env.sif

sam_3d_body_path=$bind_mnt_path/projects/sam-3d-body
input_folder=$bind_mnt_path/datasets/sam_3d_body/images_raw
output_folder=$bind_mnt_path/datasets/sam_3d_body/outputs/images_with_kp
kp_folder=$bind_mnt_path/datasets/sam_3d_body/bboxes_kps
checkpoint_path=$bind_mnt_path/large_models/sam-3d-body-dinov3/model.ckpt
mhr_path=$bind_mnt_path/large_models/sam-3d-body-dinov3/assets/mhr_model.pt

apptainer exec --nv --bind $mnt_path:$bind_mnt_path $sif_path python $sam_3d_body_path/demo.py --image_folder $input_folder --output_folder $output_folder --checkpoint_path $checkpoint_path --mhr_path $mhr_path --bbox_kp_folder $kp_folder