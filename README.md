# Tabular Data Synthesis Using GANs

This repository contains the implementations of the models used in the coursework "Tabular Data Synthesis Using Generative Models" as well as the report.

## Training

In order to train one of the models run the following command:

```bash
pyhton3 train.py -cn=MODEL_NAME
```

where `MODEL_NAME` is one of the following config names corresponding to the respective model: `gan` &ndash; Vanilla GAN, `tgan` &ndash; Table-GAN, `ctgan` &ndash; CTGAN, `ctabgan` &ndash; CTABGAN.

## Inference

In order to inference one of the models (i.e., synthesize new dataset entries) run the following command:

```bash
pyhton3 inference.py -cn=MODEL_NAME_inference
```

where `MODEL_NAME` is the name of the model from the previous section.

## Credits

This repository uses the following [project template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
