import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Tuple

class Tester(object):
    def __init__(self, p_val=0.05):
        self.p_val = p_val

    def print_results(self):
        return NotImplementedError()


class NormalityTester(Tester):
    def __init__(self, data: np.ndarray, need_plot=True, is_small=True):
        super().__init__()
        self.need_plot = need_plot
        self.is_small = is_small
        self.data = data

    def is_normal(self, p) -> bool:
        assert type(p) is float
        if p > self.p_val:
            return True
        else:
            return False

    def print_stats(self, test_name:str, stat: Tuple[float], p: Tuple[float], normality: Tuple[bool]):
        print(f"{test_name}")
        print(f"Sample1: statistic: {stat[0]}, p-value: {p[0]}, is_normal: {str(normality[0])}")
        print(f"Sample2: statistic: {stat[1]}, p-value: {p[1]}, is_normal: {str(normality[1])}")

    def draw_plot(self):
        '''Draw QQ-plot'''
        plt.figure(figsize=[5,10])
        plt.add_subplot(nrows=2)
        stats.probplot(self.data[0], dist=stats.norm, plot=plt)
        plt.subplot()
        stats.probplot(self.data[1], dist=stats.norm, plot=plt)
        plt.show()
    
    def print_results(self):
        if self.is_small:
            s_stat1, s_p1 = stats.shapiro(self.data[0])
            s_stat2, s_p2 = stats.shapiro(self.data[1])
            self.print_stats("Shapiro-Wilk Test", (s_stat1,s_stat2), (s_p1,s_p2), (self.is_normal(s_p1),self.is_normal(s_p2)))
        else:
            ks_stat1, ks_p1 = stats.kstest(self.data[0])
            ks_stat2, ks_p2 = stats.kstest(self.data[1])
            self.print_stats("Kolmogorov-Smirnov Test", (ks_stat1,ks_stat2), (ks_p1,ks_p2), (self.is_normal(ks_p1),self.is_normal(ks_p2)))

        k_stat1, k_p1 = stats.normaltest(self.data[0])
        k_stat2, k_p2 = stats.normaltest(self.data[1])
        self.print_stats("d'Agostino K-square Test",(k_stat1,k_stat2), (k_p1,k_p2), (self.is_normal(k_p1),self.is_normal(k_p2)))

        self.draw_plot()


class HypothesisTester(Tester):
    def __init__(self, data: np.ndarray, is_tost=False, normality=False, is_pair=True):
        super().__init__()
        self.normality = normality
        self.is_pair = is_pair
        self.data = data
        self.is_tost = is_tost

    def reject_alternative(self, p) -> bool:
        if p > self.p_val:
            return True
        else:
            return False 
    @staticmethod
    def truncate(p: float) -> float:
        if p < 0.05:
            return p
        string = repr(p)
        integer, decimal = string.split('.')
        if len(decimal) >= 2: 
            decimal = decimal[:2]
        return float(f'{integer}.{decimal}')

    @staticmethod
    def opposite(h_dir: str) -> str:
        if h_dir == 'less':
            return 'greater'
        elif h_dir == 'greater':
            return 'less'
        else:
            raise ValueError("Hypothesis direction must be either 'less' or 'greater' for one-sided tests!")
    
    def tost(self, diff=0.1, h_dir = 'greater'):
        assert self.data.shape[0] >= 2

        test_name = ''
        # data_0: PCMG
        # data_1: Comparison

        ### Parametric
        if self.normality and self.is_pair:
            # Paired T-test TOST (considering MWP)
            test_name = "Paired T-test TOST"
            d = self.data[1] - diff
            upper_stat, upper_p = stats.ttest_rel(self.data[0], d, alternative=h_dir)
            d = self.data[1] + diff
            lower_stat, lower_p = stats.ttest_rel(self.data[0], d, alternative=self.opposite(h_dir))

        elif self.normality:
            # T-test TOST (not considering MWP)
            test_name = "T-test TOST"
            print(self.data[1])
            d = self.data[0] - (self.data[1] - diff)
            upper_stat, upper_p = stats.ttest_1samp(d, -diff, alternative=h_dir)
            d = self.data[0] - (self.data[1] + diff)
            lower_stat, lower_p = stats.ttest_1samp(d, diff, alternative=self.opposite(h_dir))

        ### Non-parametric
        elif self.is_pair:
            # Wilcoxon signed-rank TOST (considering MWP)
            test_name = "Wilcoxon signed-rank TOST"
            upper_stat, upper_p = stats.wilcoxon(self.data[1] - diff, alternative=h_dir)
            lower_stat, lower_p = stats.wilcoxon(self.data[1] + diff, alternative=self.opposite(h_dir))

        else:
            # Mann-Whitney U-test TOST (not considering MWP)
            test_name = "Mann-Whitney U-test TOST"
            d = self.data[1] - diff
            upper_stat, upper_p = stats.mannwhitneyu(self.data[0], d, alternative=h_dir)
            d = self.data[1] + diff
            lower_stat, lower_p = stats.mannwhitneyu(self.data[0], d, alternative=self.opposite(h_dir))

        print(test_name)

        return test_name, max(upper_p.mean(), lower_p.mean())/2.


    def one_way_test(self, h_dir = 'greater'):
        if self.normality and self.is_pair:
            # Paired T-test (considering MWP)
            test_name = "Paired T-test"
            stat, p = stats.ttest_rel(self.data[0], self.data[1], alternative=h_dir)
        elif self.normality:
            # T-test (not considering MWP)
            test_name = "T-test"
            stat, p = stats.ttest_ind(self.data[0], self.data[1], alternative=h_dir)

        elif self.is_pair:
            # Wilcoxon signed-rank (considering MWP)
            test_name = "Wilcoxon signed-rank"
            d = self.data[0] - self.data[1]
            stat, p = stats.wilcoxon(d, alternative=h_dir)

        else:
            # Mann-Whitney U-test (not considering MWP)
            test_name = "Mann-Whitney U-test"
            stat, p = stats.mannwhitneyu(self.data[0], self.data[1], alternative=h_dir)
            
        print(test_name)
        return test_name, stat, p

    def print_stats_tost(self, p: float, reject_alternative: bool):
        print(f"p-value: {p}, reject_alternative: {str(reject_alternative)}")


    def print_stats_one_way(self, stat: float, p: float, reject_alternative: bool):
        print(f"statistic: {stat}, p-value: {p}, reject_alternative: {str(reject_alternative)}")
    
    
    def print_results(self):
        if self.is_tost:
            test_name, p = self.tost(self.data)
            p = self.truncate(p)
            self.print_stats_tost(p, self.reject_alternative(p))
        else:
            test_name, stat, p = self.one_way_test()
            p = self.truncate(p)
            self.print_stats_one_way(stat, p, self.reject_alternative(p))


__all__ = ['Tester','NormalityTester', 'HypothesisTester']