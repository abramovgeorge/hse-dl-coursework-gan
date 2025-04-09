# Vanilla GAN for MNIST dataset

This is the implementation of Vanilla GAN architecture for the MNIST dataset.

## Installation

1. Install all required packages:

```bash
pip install -r requirements.txt
```

2. Install `pre-commit`:

```bash
pre-commit install
```

## Training

To train the model run the following command:

```bash
pyhton3 train.py
```

## Inference

To inference the model run the following command:

```bash
pyhton3 inference.py
```

In the inference config you can specify number of generated images as well as the type (i.e., digit). Results are saved to `data/saved/gan_mnist` by default, the folder can be changed in the config.

## Credits

This repository uses the following [project template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
