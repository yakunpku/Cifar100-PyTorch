# Cifar100-PyTorch
Classification on CIFAR-100 with PyTorch.

## Train 
```shell
sh train_cifar100.sh resnety-110 0
```

## Results
|                   Model                        |      Params    |     GFlops     |    Acc Top1   |    Acc Top5    |
| ---------------------------------------------- | -------------- | -------------- | ------------- | -------------- |
|     resnet-110(Basic Block)                    |      1.74 M    |      0.26      |    69.140 %   |    90.880 %    |
|     resnet-110(Basic Block, Label Smooth)      |      1.74 M    |      0.26      |    70.410 %   |    89.810 %    |