#!/bin/bash
#SBATCH --partition=A100x8_twodays,A100x8
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/mnt/scratch/sibo/yolov5/logs/8class_%j.out
#SBATCH --error=/mnt/scratch/sibo/yolov5/logs/8class_%j.err
#SBATCH --exclude=asimov-240,asimov-241,asimov-243

# export PATH=/tools/cuda_10.0.130/bin:$PATH
# export PATH=/tools/anaconda3-5.1.0/bin:$PATH
# export NCCL_P2P_DISABLE=1
# module load cuda/11.7.0
# module load cudnn/8.6.0
# module load nccl/2.18.1
# module load python/3.8.5

cd /mnt/scratch/sibo/yolov5
source /tools/anaconda3-default/etc/profile.d/conda.sh
conda activate yolov5

srun python -m torch.distributed.run --nproc_per_node 8 train.py --data coco_chile_dino_8class.yaml --weights runs/train/exp302/weights/best.pt --batch-size 128 --freeze 10 --device 0,1,2,3,4,5,6,7 --epochs 100
