arch=$1
gpu=$2

python train.py \
--arch $arch \
--block-name Bottleneck \
--seed 0 \
--optimizer SGD \
--loss_type ce \
--num-classes 100 \
--num-epochs 164 \
--milestones 81 122 \
--batch-size 128 \
--learning-rate 0.1 \
--lr-gamma 0.1 \
--warmup-step -1 \
--momentum 0.9 \
--weight-decay 1.e-4 \
--checkpoint-cycle 10 \
--gpu $gpu 
