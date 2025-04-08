# Tabular Data Synthesis Using GANs

This repository contains the implementations of the models used in the coursework "Tabular Data Synthesis Using Generative Models" as well as the [report](coursework.pdf).

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

To train one of the models run the following command:

```bash
pyhton3 train.py -cn=MODEL_NAME
```

where `MODEL_NAME` is one of the following config names corresponding to the respective model: `gan` &ndash; Vanilla GAN, `tgan` &ndash; Table-GAN, `ctgan` &ndash; CTGAN, `ctabgan` &ndash; CTABGAN.

## Inference

To inference one of the models (i.e., synthesize new dataset entries) run the following command:

```bash
pyhton3 inference.py -cn=MODEL_NAME_inference
```

where `MODEL_NAME` is the name of the model from the previous section.

In the corresponding inference config you can specify number of generated entries for each class, or, in the case of conditional models (CTGAN, CTABGAN), target discrete feature. The results are saved to `data/saved/MODEL_NAME` folder, which also can be specified in the config.

Metrics for the report were calculated using this [script](scripts/metrics.py).

## Credits

This repository uses the following [project template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
