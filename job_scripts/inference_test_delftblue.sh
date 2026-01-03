#!/bin/bash
#SBATCH --job-name="inference_test_delftblue"
#SBATCH --partition=gpu-a100 # Request partition. Default is 'general' 
#SBATCH --time=2:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=6    
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --account=innovation
#SBATCH --output=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/zli33/slurm_outputs/sam_4d_body/slurm_%j.err # Set name of error log. %j is the Slurm jobId


home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/sam4d_checkpoints
data_path=$scratch_path/data/sam4d_body

bind_model_path=/mnt/sam4d_checkpoints
bind_data_path=/mnt/data/sam4d_body
bind_home_path=/mnt/home/zli33

sif_path=$scratch_path/apptainers/body4d_osmesa.sif
project_folder=$bind_home_path/projects/sam-body4d
input_folder=$bind_data_path/inputs
output_folder=$bind_data_path/outputs

# Make a unique experiment subfolder to avoid overwriting prior runs
timestamp=$(date +%Y%m%d_%H%M%S)
exp_dir=$output_folder/exp_$timestamp
mkdir -p $exp_dir

# apptainer exec --nv --bind $neon_path:$bind_neon_path --bind $zli_path:$bind_zli_path $sif_path python $project_folder/infer_video.py --video $input_folder/cam04_cut_03.mp4 --output $output_folder 

apptainer exec --nv \
  --bind $model_path:$bind_model_path \
  --bind $data_path:$bind_data_path \
  --bind $home_path:$bind_home_path \
  --env PYTHONPATH=$project_folder/models/sam3:$project_folder:$PYTHONPATH \
  --env PYOPENGL_PLATFORM=osmesa \
  $sif_path \
  python $project_folder/infer_video.py --video $input_folder/cam04_cut_03.mp4 --output $exp_dir

# apptainer exec --env PYOPENGL_PLATFORM=osmesa $sif_path python -c "from OpenGL.osmesa import OSMesaCreateContextAttribs; print('ok')"