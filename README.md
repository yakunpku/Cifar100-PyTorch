# Cifar100-PyTorch
Classification on CIFAR-100 with PyTorch.

## Train 
```shell
sh train_cifar100.sh resnety-110 0
```

## Evaluate
```shell
python evaluate.py --arch resnet-110 --block-name bottleneck --checkpoint-path experiments/resnet-110/checkpoint_best.pth --gpu 2
```

## Results
|                   Model                        |      Params    |     GFlops     |    Acc Top1   |    Acc Top5    |
| ---------------------------------------------- | -------------- | -------------- | ------------- | -------------- |
|   resnet-110(Basic Block)                      |      1.74 M    |      0.26      |    69.140 %   |    90.880 %    |
|   resnet-110(Basic Block, Label Smooth)        |      1.74 M    |      0.26      |    70.410 %   |    89.810 %    |
|   resnet-110(Bottleneck Block)                 |      1.17 M    |      0.17      |    72.560 %   |    92.340 %    |
|   resnet-110(Bottleneck Block, Label Smooth)   |      1.17 M    |      0.17      |    71.370 %   |    90.810 %    |
|   resnet-110(Bottleneck Block, Warmup Step 5)  |      1.17 M    |      0.17      |    72.250 %   |    92.430 %    |
|   resnet-110(Bottleneck Block, cosine_lr)  |      1.17 M    |      0.17      |    72.250 %   |    92.430 %    |