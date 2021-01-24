arch=$1
gpu=$2

python train.py \
--arch $arch \
--seed 10007 \
--num-classes 100 \
--milestones 81 122 \
--num-epochs 170 \
--batch-size 128 \
--learning-rate 0.1 \
--lr-gamma 0.1 \
--momentum 0.9 \
--weight-decay 1.0e-4 \
--checkpoint-cycle 10 \
--gpu $gpu
