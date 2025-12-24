#!/bin/bash
#SBATCH --partition=insy,general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=2:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --mem=8G
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=/home/nfs/zli33/slurm_outputs/sam_3d_body/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/nfs/zli33/slurm_outputs/sam_3d_body/slurm_%j.err # Set name of error log. %j is the Slurm jobId

#SBATCH --gres=gpu:a40:1 # Request 1 GPU
neon_path=/tudelft.net/staff-umbrella/neon
zli_path=/home/nfs/zli33
apptainer_path=/tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan

bind_neon_path=/mnt/neon
bind_zli_path=/mnt/zli33
bind_apptainer_path=/mnt/zonghuan

sif_path=$bind_apptainer_path/large_builds/containers/sam-body4d.sif
project_folder=$bind_zli_path/projects/sam-body4d
input_folder=$bind_neon_path/zonghuan/data/sam4d_body/inputs
output_folder=$bind_neon_path/zonghuan/data/sam4d_body/outputs

apptainer exec --nv --bind $neon_path:$bind_neon_path --bind $zli_path:$bind_zli_path --bind $apptainer_path:$bind_apptainer_path $sif_path python $project_folder/infer_video.py --video $input_folder/cam04_cut_03.mp4 --output $output_folder 

# bulk_path=/tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan
# local_path=/home/nfs/zli33
# bind_bulk_path=/mnt/zonghuan
# bind_local_path=/mnt/zli33
# sif_path=$bulk_path/large_builds/containers/detectron_env.sif

# sam_3d_body_path=$bind_local_path/projects/sam-3d-body
# input_folder=$bind_bulk_path/datasets/sam_3d_body/images_test
# output_folder=$bind_bulk_path/datasets/sam_3d_body/outputs/images_test
# kp_folder=$bind_bulk_path/datasets/sam_3d_body/bboxes_kps
# checkpoint_path=$bind_bulk_path/large_models/sam-3d-body-dinov3/model.ckpt
# mhr_path=$bind_bulk_path/large_models/sam-3d-body-dinov3/assets/mhr_model.pt

# apptainer exec --nv --bind $bulk_path:$bind_bulk_path --bind $local_path:$bind_local_path $sif_path python $sam_3d_body_path/demo.py --image_folder $input_folder --output_folder $output_folder --checkpoint_path $checkpoint_path --mhr_path $mhr_path --bbox_kp_folder $kp_folder