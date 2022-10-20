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
    def __init__(self, need_plot=True, is_small=True):
        super().__init__()
        self.need_plot = need_plot
        self.is_small = is_small

    def is_normal(self, p) -> bool:
        if p < self.p_val:
            return True
        else:
            return False

    def print_stats(self, test_name:str, stat: float, p: float, normality: bool):
        print(f"{test_name}")
        print(f"statistic: {stat}, p-value: {p}, is_normal: {str(normality)}")

    def draw_plot(self):
        '''Draw QQ-plot'''
        plt.figure(5,5)
        stats.probplot(self.data, dist=stats.norm, plot=plt)
        plt.show()
    
    def print_results(self):
        if self.is_small:
            ks_stat, ks_p = stats.shapiro(self.data, nan_policy='omit')
            self.print_stats("Shapiro-Wilk Test", ks_stat, ks_p, self.is_normal(ks_p))
        else:
            ks_stat, ks_p = stats.normaltest(self.data, nan_policy='omit')
            self.print_stats("Kolmogorov-Smirnov Test",ks_stat, ks_p, self.is_normal(ks_p))

        k_stat, k_p = stats.kstest(self.data, 'norm')
        self.print_stats("d'Agostino K-square Test",k_stat, k_p, self.is_normal(k_p))

        self.draw_plot()


class HypothesisTester(Tester):
    def __init__(self, normality=True, is_pair=True):
        super().__init__()
        self.nomality = normality
        self.is_pair = is_pair
    
    @staticmethod
    def opposite(h_dir: str) -> str:
        if h_dir == 'less':
            return 'greater'
        elif h_dir == 'greater':
            return 'less'
        else:
            raise ValueError("Hypothesis direction must be either 'less' or 'greater' for one-sided tests!")
    
    def tost(self, diff=0.05):
        assert self.data.shape[0] >= 2
        h_dir = 'greater'

        # data_0: PCMG
        # data_1: Comparison

        ### Parametric
        if self.normality and self.is_pair:
            # Paired T-test TOST (considering MWP)
            data_1 = self.data[1] - diff
            upper_stat, upper_p = stats.ttest_rel(self.data[0], data_1, alternative=h_dir)
            data_1 = self.data[1] + diff
            lower_stat, lower_p = stats.ttest_rel(self.data[0], data_1, alternative=self.opposite(h_dir))

        elif self.normality:
            # T-test TOST (not considering MWP)
            data_1 = self.data[1] - diff
            upper_stat, upper_p = stats.ttest_ind(self.data[0], data_1, alternative=h_dir)
            data_1 = self.data[1] + diff
            lower_stat, lower_p = stats.ttest_ind(self.data[0], data_1, alternative=self.opposite(h_dir))

        ### Non-parametric
        elif not self.is_pair:
            # Wilcoxon signed-rank TOST (considering MWP)
            d = self.data[0] - (self.data[1] - diff)
            upper_stat, upper_p = stats.wilcoxon(d, alternative=h_dir)
            d = self.data[0] - (self.data[1] + diff)
            lower_stat, lower_p = stats.wilcoxon(d, alternative=self.opposite(h_dir))

        else:
            # Mann-Whitney U-test TOST (not considering MWP)
            data_1 = self.data[1] - diff
            upper_stat, upper_p = stats.mannwhitneyu(self.data[0], data_1, alternative=h_dir)
            data_1 = self.data[1] + diff
            lower_stat, lower_p = stats.mannwhitneyu(self.data[0], data_1, alternative=self.opposite(h_dir))

        return upper_stat, upper_p, lower_stat, lower_p


    def one_way_test(self):
        if self.normality and self.is_pair:
            # Paired T-test (considering MWP)
            stat, p = stats.ttest_rel()
        elif self.normality:
            # T-test (not considering MWP)
            stat, p = stats.ttest_ind()

        elif not self.is_pair:
            # Wilcoxon signed-rank (considering MWP)
            stat, p = stats.wilcoxon()

        else:
            # Mann-Whitney U-test (not considering MWP)
            stat, p = stats.mannwhitneyu()

        return stat, p
        

__all__ = ['NormalityTester', 'HypothesisTester']