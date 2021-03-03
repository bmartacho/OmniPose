# CFG='experiments/mpii/hrnet/w48_256x256_adam_lr1e-3.yaml'

# CFG='experiments/coco/hrnet/w48_128x96_adam_lr1e-3.yaml'
# CFG='experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml'
CFG='experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
# CFG='experiments/coco/hrnet/w48_512x384_adam_lr1e-3.yaml'
# CFG='experiments/coco/hrnet/w48_640x640_adam_lr1e-3.yaml'

# CFG='experiments/posetrack/omnipose_384x288.yaml'

OPTS=None
MODELDIR=''
LOGDIR=''
DATADIR=''
PREVMODELDIR=''

python train.py \
  --cfg="$CFG" \
  --opts="$OPTS" \
  --modelDir="$MODELDIR" \
  --logDir="$LOGDIR" \
  --dataDir="$DATADIR" \
  --prevModelDir="$PREVMODELDIR"