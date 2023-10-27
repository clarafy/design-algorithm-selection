import hydra
from omegaconf import DictConfig, OmegaConf
from time import time
from datetime import datetime
from pytz import timezone
# import wandb

import numpy as np
import torch

import shifts
import tdre

tz_time = datetime.now(timezone('America/Los_Angeles'))
OmegaConf.register_new_resolver('nowpt', lambda pattern: tz_time.strftime(pattern))
@hydra.main(config_path="./hydra_config", config_name="gaussians")
def main(cfg: DictConfig):
    dtype = torch.double if cfg.dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.")

    # wandb.init(project='dre', group='gaussians', entity="prescient-design")

    ds = np.logspace(np.log2(cfg.d_min), np.log2(cfg.d_max), cfg.n_d, base=2).astype(int)
    ds = [d if d % 2 == 0 else d + 1 for d in ds]
    ms = np.linspace(cfg.m_min, cfg.m_max, cfg.n_m).astype(int)
    print('Dimensions: {}'.format(ds))
    print('Numbers of ratios: {}'.format(ms))
    t0 = time()
    for m in ms:
        for d in ds:

            # get samples from p0 (target, numerator) and pm (source, denominator)
            gaussians = shifts.Gaussians(d)
            Xm_nxd, X0_nxd = gaussians.get_data(cfg.n)

            # fit bridges
            model = tdre.UnsharedTelescopingLogDensityRatioEstimator(
                tdre.Quadratic, d, n_ratio=m, device=device, dtype=dtype)
            config = {
                'val_frac': cfg.optimizer.val_frac,
                'n_steps': cfg.optimizer.n_steps,
                'lr': cfg.optimizer.lr
            }
            train_dfs = model.fit(X0_nxd, Xm_nxd, config)

            # predict log density ratios on fresh data
            Xm_nxd, _ = gaussians.get_data(cfg.n)
            ldrpred_n = model.predict_log_dr(Xm_nxd)
            ldr_n = gaussians.get_log_dr(Xm_nxd)
            rmse = np.sqrt(np.mean(np.square(ldr_n - ldrpred_n)))

            # save results
            np.savez(
                'm{}-d{}.npz'.format(m, d),
                ldrpred_n=ldrpred_n,
                ldr_n=ldr_n
            )
            for i in range(m):
                train_dfs[i].to_csv('train-{}m{}-d{}.csv'.format(i, m, d))
            print('Done with m = {}, d = {} (RMSE = {:.3f}, {} s).'.format(m, d, rmse, int(time() - t0)))
    # wandb.finish()

if __name__ == "__main__":
    main()

