# Design algorithm selection
Repo accompanying the ICML 2025 submission "Reliable algorithm selection for machine learning-guided design".

See gb1-demo.ipynb for how to reproduce the algorithm selection experiments for designing protein GB1 (Fig. 3 in the main text). Descriptions of other files are as follows:
- `gb1.py`: main functions for GB1 experiments
- `utils.py`: utilities and other methods (GMMForecasts, CalibratedForecasts, and conformal prediction methods)
- `models.py`: classes for predictive models
- `designers.py`: design algorithms used in RNA binder experiments
- `vae.py`: VAE used by CbAS and DbAS in designers.py
- `dre.py`: classes for density ratio estimation used in RNA binder experiments