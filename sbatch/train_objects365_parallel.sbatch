#!/bin/bash
#SBATCH --partition=V100x8,V100x8_sixdays,A100x8_twodays
#SBATCH --gres=gpu:8
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --output=/mnt/scratch/sibo/yolov5/logs/Objects365_%j.out
#SBATCH --error=/mnt/scratch/sibo/yolov5/logs/Objects365_%j.err
#SBATCH --exclude=

cd /mnt/scratch/sibo/yolov5
source /tools/anaconda3-default/etc/profile.d/conda.sh
conda activate yolov5

# srun python -m torch.distributed.run --nproc_per_node 8 train.py --data Objects365.yaml --weights yolov5m.pt --batch-size 128 --device 0,1,2,3,4,5,6,7 --epochs 150

srun python -m torch.distributed.run --nproc_per_node 8 --nnodes 3 --node_rank 0 --master_addr "192.168.1.1" --master_port 1234 train.py --batch-size 128 --data Objects365.yaml  --device 0,1,2,3,4,5,6,7 --epochs 150
