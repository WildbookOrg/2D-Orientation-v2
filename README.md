# 2D-Orientation-v2
Refactor and improvements from the original 2D orientation project

python train.py -h

The program runs as follows:
  
  if(example or save-all-figs):
    do example
  elif(test):
    do test
  else:
    train

### Option Descriptions
--type: used to specifify a unique name for different optimizations that arent uncluded in progress folder naming, otherwize it would write over a state dict that has different optimization parameters but the same name
  options used to name progress folder:
    animal
    type
    nClasses
    pretrain

--no-resume: clean state dict and start training from scratch

--pretrain: use pretrained weights from official pytorch densenet

--separate-trig: instead of estimating the angle theta, the network has two outputs corresponding to cos and sin of the angle theta, and uses arctan2 to obtain theta. This is important because it changes nClasses to 2 instead of 1. 

--degree loss: only useful when determining if estimating the angle in degrees or radians increases accuracy


### Example Usage
#### Show an Example of Current Estimation: 
python3 train.py --type regression --nClasses 2 --device 0 --separate-trig --batchSz 3 --animal seadragon --example

#### Training (using slurm): 
srun --time=240 --gres=gpu:1 --ntasks=1 python3 train.py --type regression --nClasses 2 --device 0 --separate-trig --batchSz 50 --animal seadragon

#### Show Loss History: 
python3 train.py --type regression --nClasses 2 --separate-trig --pretrain --plot-loss-history --animal seadragon
