arch=$1
gpu=$2

python train.py \
--arch $arch \
--seed 0 \
--optimizer Adam \
--num-classes 100 \
--milestones 60 120 160 \
--num-epochs 200 \
--batch-size 32 \
--learning-rate 0.1 \
--lr-gamma 0.1 \
--warmup-step -1 \
--momentum 0.9 \
--weight-decay 5.e-4 \
--checkpoint-cycle 10 \
--gpu $gpu
