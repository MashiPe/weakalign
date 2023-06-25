## About

This is an extesion that adds support for the homography transformation for the model proposed in "End-to-end weakly-supervised semantic alignment" by I. Rocco, R. Arandjelović and J. Sivic. 

Credit for the authors. For more information check out the original project [[website](http://www.di.ens.fr/willow/research/weakalign/)] and the paper on [[arXiv](https://arxiv.org/abs/1712.06861)].

## Getting started

### VScode dev-container

A dev docker container is provided for easy reproduction. To use it open the project in VSCode and open the project in a dev-container using the Dockerfile located at the root of the project. 

### Training

The code includes scripts for pre-training the homography model with strong supervision (`train_strong_hom.py`) as well as to fine-tune the model using weak supervision (`train_weak_hom.py`).

Training scripts can be found in the `scripts/` folder.

### Evaluation

Evaluation is implemented in the `eval_hom.py` file. It can evaluate a homography model (with the `--model-hom` parameters). No evaluation for combined models is provided

The evaluation dataset is passed with the `--eval-dataset` parameter.

### Trained models

Trained models for the baseline method using only strong supervision and the proposed method using additional weak supervision are provided below. You can store them in the `trained_models/` folder. 

**CNNGeometric with ResNet-101 baseline:** [[hom model]()]

```
python eval_hom.py --feature-extraction-cnn resnet101 --model-hom trained_models/cnngeo_resnet101_hom.pth.tar --eval-dataset pf-pascal
```

**Proposed method:** [[hom model]()]

```
python eval_hom.py --feature-extraction-cnn resnet101 --model-hom trained_models/weakalign_resnet101_hom.pth.tar --eval-dataset pf-pascal
```

## Credits

If you want a more in depth view of the method check the original work.

