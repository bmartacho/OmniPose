# CFG='experiments/mpii/hrnet/w48_256x256_adam_lr1e-3.yaml'
CFG='experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
OPTS=None
MODELDIR=''
LOGDIR=''
DATADIR=''
PREVMODELDIR=''

python test.py \
  --cfg="$CFG" \
  --opts="$OPTS" \
  --modelDir="$MODELDIR" \
  --logDir="$LOGDIR" \
  --dataDir="$DATADIR" \
  --prevModelDir="$PREVMODELDIR"


# python inference.py \
#   --cfg="$CFG" \
#   --opts="$OPTS" \
#   --modelDir="$MODELDIR" \
#   --logDir="$LOGDIR" \
#   --dataDir="$DATADIR" \
#   --prevModelDir="$PREVMODELDIR"