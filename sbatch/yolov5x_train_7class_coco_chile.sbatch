#!/bin/bash
#SBATCH --partition=V100x8_sixdays,V100x8,A100x8_twodays,A100x8
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=/mnt/scratch/sibo/yolov5/logs/yolov5l_coco_chile_7class_%j.out
#SBATCH --error=/mnt/scratch/sibo/yolov5/logs/yolov5l_coco_chile_7class_%j.err
#SBATCH --exclude=asimov-243

cd /mnt/scratch/sibo/yolov5
source /tools/anaconda3-default/etc/profile.d/conda.sh
conda activate yolov5

srun python -m torch.distributed.run --nproc_per_node 8 train.py --data coco_chile_7class.yaml --weights yolov5x.pt --batch-size 128 --device 0,1,2,3,4,5,6,7 --epochs 100 --patience 5 --hyp hyp.scratch-high.yaml
