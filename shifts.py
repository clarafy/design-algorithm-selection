from abc import ABC, abstractmethod

import numpy as np
import scipy as sc


class DistributionShift(ABC):
    def __init__(self, d: int):
        self.d = d

    @abstractmethod
    def get_data(self, n: int):
        raise NotImplementedError

    @abstractmethod
    def get_log_dr(self, X_nxd: np.array):
        raise NotImplementedError

class DiscreteSequences(DistributionShift):
    def __init__(self, d, p: float = 0.1):
        self.d = d
        self.p = p
        self.logp = np.log(p)
        self.log1minp = np.log(1 - p)
        self.c_d = np.random.binomial(1, 0.5, size=d)

    def sample(self, distribution: str):
        flip_d = np.random.binomial(1, self.p, size=self.d)
        if distribution == "source":
            return np.array([1 - b if f else b for b, f in zip(self.c_d, flip_d)])
        elif distribution == "target":
            return np.array([b if f else 1 - b for b, f in zip(self.c_d, flip_d)])
        else:
            raise ValueError('Unknown distribution type: {}'.format(distribution))

    def get_data(self, n: int):
        X0_nxd = np.vstack([self.sample('target')[None, :] for _ in range(n)])
        Xm_nxd = np.vstack([self.sample('source')[None, :] for _ in range(n)])
        return Xm_nxd, X0_nxd

    def get_log_dr(self, X_nxd: np.array):
        # source distribution
        flips_n = np.sum(X_nxd != self.c_d[None, :], axis=1, keepdims=False)
        logq_n = flips_n * self.logp + (self.d - flips_n) * self.log1minp

        # target distribution
        logp_n = flips_n * self.log1minp + (self.d - flips_n) * self.logp
        return logp_n - logq_n


class Gaussians(DistributionShift):
    def __init__(self, d, mu: float = 0):
        if d % 2 != 0:
            raise ValueError('Number of dimensions {} is not even.'.format(d))
        super().__init__(d)

        # denominator/source distribution
        self.q_mean = np.zeros([self.d])
        self.q_cov = np.eye(self.d)

        # numerator/target distribution
        self.mu = mu
        self.p_mean = mu * np.ones([self.d])
        self.p_cov = self.get_cov()

    def get_cov(self):
        cov = np.eye(self.d)
        for i in range(0, self.d, 2):
            cov[i, i + 1] = 0.8
            cov[i + 1, i] = 0.8
        return cov

    def get_data(self, n: int):
        Xq_nxd = sc.stats.multivariate_normal.rvs(mean=self.q_mean, cov=self.q_cov, size=n)
        Xp_nxd = sc.stats.multivariate_normal.rvs(mean=self.p_mean, cov=self.p_cov, size=n)
        return Xq_nxd, Xp_nxd

    def get_log_dr(self, X_nxd: np.array):
        if X_nxd.shape[1] != self.d:
            raise ValueError('Dimension {} does not equal class dimension {}'.format(X_nxd.shape[1], self.d))
        logd_n = sc.stats.multivariate_normal.logpdf(X_nxd, mean=self.q_mean, cov=self.q_cov, )
        logn_n = sc.stats.multivariate_normal.logpdf(X_nxd, mean=self.p_mean, cov=self.p_cov)
        return logn_n - logd_n
