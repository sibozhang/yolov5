#!/bin/bash
#SBATCH --partition=V100x8,V100x8_sixdays,A100x8_twodays,A100x8,V100_DGX
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/mnt/scratch/sibo/yolov5/logs/2class_%j.out
#SBATCH --error=/mnt/scratch/sibo/yolov5/logs/2class_%j.err
#SBATCH --exclude=asimov-245,asimov-246

cd /mnt/scratch/sibo/yolov5
source /tools/anaconda3-default/etc/profile.d/conda.sh
conda activate yolov5

srun python -m torch.distributed.run --nproc_per_node 8 train.py --data coco_chile_dino_cabin.yaml --weights runs/train/exp372/weights/last.pt --batch-size 128 --freeze 10 --device 0,1,2,3,4,5,6,7 --epochs 100
