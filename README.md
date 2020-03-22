# 2D-Orientation-v2
Refactor and improvements from the original 2D orientation project

python train.py -h

example:

python3 train.py --type regression --nClasses 2 --device 0 --separate-trig --batchSz 3 --animal seadragon --example

training:

srun --time=240 --gres=gpu:1 --ntasks=1 python3 train.py --type regression --nClasses 2 --device 0 --separate-trig --batchSz 50 --animal seadragon

loss history:

python3 train.py --type regression --nClasses 2 --separate-trig --pretrain --plot-loss-history --animal seadragon
