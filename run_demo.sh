# # CFG='experiments/mpii/hrnet/w48_256x256_adam_lr1e-3.yaml'
# CFG='experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'
# OPTS=None
# MODELDIR=''
# LOGDIR=''
# DATADIR=''
# PREVMODELDIR=''

# python demo.py \
#   --cfg="$CFG" \
#   --opts="$OPTS" \
#   --modelDir="$MODELDIR" \
#   --logDir="$LOGDIR" \
#   --dataDir="$DATADIR" \
#   --prevModelDir="$PREVMODELDIR"

# python demo_samples.py \
# 	--dataset COCO \
#     --prediction output/coco/omnipose/OmniPose_HRw48_v2/results/keypoints_val2017_results_0.json \
#     --save-path visualization/coco/

# python demo_samples.py \
# 	--dataset MPII \
#     --prediction output/mpii/omnipose/OmniPose_HRw48_v2/pred.mat \
#     --save-path visualization/mpii/