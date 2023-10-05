#!/bin/bash
#SBATCH --partition=V100x8_sixdays,A100x8_twodays,V100x8
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/mnt/scratch/sibo/yolov5/logs/Objects365_chile_dino_8class_%j.out
#SBATCH --error=/mnt/scratch/sibo/yolov5/logs/Objects365_chile_dino_8class_%j.err
#SBATCH --exclude=

cd /mnt/scratch/sibo/yolov5
source /tools/anaconda3-default/etc/profile.d/conda.sh
conda activate yolov5

srun python -m torch.distributed.run --nproc_per_node 8 train.py --data Objects365_chile_dino_8class.yaml --weights runs/train/exp376/weights/best.pt --batch-size 128 --device 0,1,2,3,4,5,6,7 --epochs 200 --cache
