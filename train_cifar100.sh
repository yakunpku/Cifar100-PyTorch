python train.py \
--which-model resnet18 \
--seed 0 \
--num-classes 100 \
--milestones 60 120 160 \
--num-epochs 200 \
--batch-size 128 \
--learning-rate 0.1 \
--lr-gamma 0.2 \
--momentum 0.9 \
--weight-decay 5.0e-4 \
--checkpoint-cycle 10 \
--gpu 2
