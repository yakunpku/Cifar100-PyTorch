import os
import functools
import logging


port = 85

candidate_num = 10
test_embeddings = 'static/embeddings/test_embeddings.list'
test_image_dir = "static/test/images"
test_image_list = "static/test/image.list"

checkpoint_path = "/data/toolkits/Cifar100-PyTorch/experiments/resnet-110/checkpoint_best.pth"
device = "cuda:0"