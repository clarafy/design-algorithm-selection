# Design algorithm selection
This repo accompanies the paper "Reliable algorithm selection for machine learning-guided design" (ICML 2025).

## Installation

The notebooks `gb1-demo.ipynb` and `rna-demo.ipynb` demonstrate how to reproduce the design algorithm selection experiments for protein GB1
(Fig. 3) and RNA binders (Fig. 4), respectively, in the main text. To run these, first install the necessary packages:

```
python -m venv das
source das/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Descriptions of other files are as follows:
- `gb1.py`: main functions for GB1 experiments
- `rna.py`: main functions for RNA experiments
- `utils.py`: utilities and other methods (GMMForecasts, CalibratedForecasts, conformal prediction methods)
- `models.py`: classes for predictive models
- `designers.py`: design algorithms used in RNA binder experiments
- `vae.py`: VAE used by CbAS and DbAS in designers.py
- `dre.py`: classes for density ratio estimation used in RNA binder experiments

## Acknowledgments

The RNA binder experiments use the [ViennaRNA](https://www.tbi.univie.ac.at/RNA/#) package, developed at the Institute for Theoretical Chemistry of the University of Vienna.

Lorenz, Ronny and Bernhart, Stephan H. and Höner zu Siederdissen, Christian and Tafer, Hakim and Flamm, Christoph and Stadler, Peter F. and Hofacker, Ivo L.
ViennaRNA Package 2.0
Algorithms for Molecular Biology, 6:1 26, 2011, doi:10.1186/1748-7188-6-26