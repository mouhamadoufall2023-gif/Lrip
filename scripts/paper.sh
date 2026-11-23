#!/usr/bin/env bash
# Commands for the LRIP experiments of the paper (Sec. 5). Run from the repo root.
# Each image / text experiment is repeated with seeds 1, 2, 3 and averaged.
# Swap `loss=lrip` for `loss=ce` or `loss=ls` to train the baselines.
set -e
SEEDS=${SEEDS:-"1 2 3"}

cifar10() {  # ResNet-50 (or resnet101_cifar), 150 epochs, decay at 50/100 -> t* = 50
  for s in $SEEDS; do
    python tools/train_net.py seed=$s \
      data=cifar10 model=resnet50_cifar loss=lrip loss.epsilon=0.1 \
      optim=sgd optim.lr=0.1 scheduler=multi_step "scheduler.milestones=[50,100]" \
      train.max_epoch=150
  done
}

tiny_imagenet() {  # ResNet-50 (or resnet101_tiny), 100 epochs, decay at 40/60 -> t* = 40
  for s in $SEEDS; do
    python tools/train_net.py seed=$s \
      data=tiny_imagenet model=resnet50_tiny model.num_classes=200 loss=lrip loss.epsilon=0.1 \
      optim=sgd optim.lr=0.1 scheduler=multi_step "scheduler.milestones=[40,60]" \
      train.max_epoch=100
  done
}

cub() {  # ResNet-101 (ImageNet-pretrained), OneCycle, 160 epochs, pct_start=0.3 -> t* = 48
  for s in $SEEDS; do
    python tools/train_net.py seed=$s \
      data=cub data.batch_size=16 model=resnet101 model.num_classes=200 loss=lrip loss.epsilon=0.1 \
      optim=sgd optim.lr=0.01 scheduler=one_cycle scheduler.pct_start=0.3 \
      train.max_epoch=160
  done
}

newsgroups() {  # GPCN + GloVe, Adam, OneCycle, 30 epochs, pct_start=0.3 -> t* = 9
  for s in $SEEDS; do
    python tools/train_net.py seed=$s task=nlp \
      data=newgroups data.batch_size=64 model=global_pooling loss=lrip loss.epsilon=0.3 \
      optim=adam optim.lr=0.001 scheduler=one_cycle scheduler.pct_start=0.3 \
      train.max_epoch=30
  done
}

voc() {  # DeepLabV3 (ResNet-50, ImageNet), 100 epochs, decay at 40/80 -> t* = 40
  python tools/train_net.py task=segment \
    data=voc model=deeplabv3 loss=lrip loss.epsilon=0.1 loss.ignore_index=255 \
    optim=sgd optim.lr=0.01 scheduler=multi_step "scheduler.milestones=[40,80]" \
    train.max_epoch=100
}

ablation_sd() {  # Table 5: priors taken at the second LR decay instead of the peak
  python tools/train_net.py data=cifar10 model=resnet50_cifar loss=lrip loss.snapshot_epoch=100 \
    optim=sgd optim.lr=0.1 scheduler=multi_step "scheduler.milestones=[50,100]" train.max_epoch=150
  python tools/train_net.py data=tiny_imagenet model=resnet50_tiny model.num_classes=200 \
    loss=lrip loss.snapshot_epoch=60 \
    optim=sgd optim.lr=0.1 scheduler=multi_step "scheduler.milestones=[40,60]" train.max_epoch=100
}

"${@:-cifar10}"
