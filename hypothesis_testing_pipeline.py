import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Tester(object):
    def __init__(self, data: np.array, p_val=0.05):
        self.p_val = p_val
        self.data = data
    
    def print_stats(self):
        return NotImplementedError

    def print_results(self):
        return NotImplementedError


class NormalityTester(Tester):
    def __init__(self, need_plot=True):
        super().__init__()
        self.need_plot = True

    def is_normal(self, p) -> bool:
        if p < self.p_val:
            return True
        else:
            return False

    def print_stats(self, stat: float, p: float, normality: bool):
        print(f"statistic: {stat}, p-value: {p}, is_normal: {str(normality)}")

    def draw_plot(self):
        '''Draw QQ-plot'''
        plt.figure(5,5)
        stats.probplot(self.data, dist=stats.norm, plot=plt)
        plt.show()
    
    def print_results(self):
        ks_stat, ks_p = stats.normaltest(self.data, nan_policy='omit')
        print("Kolmogorov-Smirnov Test\n")
        self.print_stats(ks_stat, ks_p, self.is_normal(ks_p))

        k_stat, k_p = stats.kstest(self.data, 'norm')
        print("d'Agostino K-square Test\n")
        self.print_stats(k_stat, k_p, self.is_normal(k_p))

        self.draw_plot()


class HypothesisTester(Tester):
    def __init__(self, h_dir:str, normality=True, is_pair=True):
        super().__init__()
        self.h_dir = h_dir
        self.nomality = normality
        self.is_pair = is_pair
    
    def tost(self, diff=0.05):
        if self.normality:
            
        else:

        

__all__ = ['NormalityTester', 'HypothesisTester']