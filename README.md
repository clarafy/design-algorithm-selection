# Design algorithm selection
Repo accompanying the paper "Reliable algorithm selection for machine learning-guided design" (ICML 2025).

See `gb1-demo.ipynb` and `rna-demo.ipynb` for how to reproduce the design algorithm selection experiments for protein GB1
(Fig. 3) and RNA binders (Fig. 4), respectively, in the main text.  Descriptions of other files are as follows:
- `gb1.py`: main functions for GB1 experiments
- `rna.py`: main functions for RNA experiments
- `utils.py`: utilities and other methods (GMMForecasts, CalibratedForecasts, conformal prediction methods)
- `models.py`: classes for predictive models
- `designers.py`: design algorithms used in RNA binder experiments
- `vae.py`: VAE used by CbAS and DbAS in designers.py
- `dre.py`: classes for density ratio estimation used in RNA binder experiments