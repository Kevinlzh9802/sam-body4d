#!/bin/bash
#SBATCH --partition=insy,general # Request partition. Default is 'general' 
#SBATCH --qos=medium         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=34:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --mem=64G
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=/home/nfs/zli33/slurm_outputs/sam_3d_body/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/nfs/zli33/slurm_outputs/sam_3d_body/slurm_%j.err # Set name of error log. %j is the Slurm jobId

#SBATCH --gres=gpu:a40:1 # Request 1 GPU
#SBATCH --gres=gpu:a40:1 # Request 1 GPU
neon_path=/tudelft.net/staff-umbrella/neon
zli_path=/home/nfs/zli33

bind_neon_path=/mnt/neon
bind_zli_path=/mnt/zli33

sif_path=$neon_path/apptainer/body4d_osmesa.sif
project_folder=$bind_zli_path/projects/sam-body4d
input_folder=$bind_neon_path/zonghuan/data/sam4d_body/inputs
output_folder=$bind_neon_path/zonghuan/data/sam4d_body/outputs

apptainer exec --nv --bind $neon_path:$bind_neon_path --bind $zli_path:$bind_zli_path $sif_path python $project_folder/infer_video.py --video $input_folder/cam04_cut_03.mp4 --output $output_folder 