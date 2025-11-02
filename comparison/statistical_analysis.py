"""
Statistical Analysis Module
Performs statistical tests and comparisons
Developed by: Student 6
"""

import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, friedmanchisquare


class StatisticalAnalysis:
    """
    StatisticalAnalysis class for comparing feature selection methods
    Performs various statistical tests
    """
    
    def __init__(self):
        """Initialize StatisticalAnalysis"""
        self.test_results = {}
    
    def paired_t_test(self, scores1, scores2, method1_name='Method 1', method2_name='Method 2'):
        """
        Paired t-test to compare two methods
        
        Args:
            scores1: Scores from method 1
            scores2: Scores from method 2
            method1_name: Name of method 1
            method2_name: Name of method 2
            
        Returns:
            dict: Test results
        """
        t_statistic, p_value = ttest_ind(scores1, scores2)
        
        result = {
            'method1': method1_name,
            'method2': method2_name,
            'method1_mean': np.mean(scores1),
            'method2_mean': np.mean(scores2),
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'better_method': method1_name if np.mean(scores1) > np.mean(scores2) else method2_name
        }
        
        self.test_results['paired_t_test'] = result
        
        print(f"\nPaired T-Test: {method1_name} vs {method2_name}")
        print(f"  {method1_name} mean: {result['method1_mean']:.4f}")
        print(f"  {method2_name} mean: {result['method2_mean']:.4f}")
        print(f"  t-statistic: {t_statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if result['significant'] else 'No'}")
        print(f"  Better method: {result['better_method']}")
        
        return result
    
    def wilcoxon_test(self, scores1, scores2, method1_name='Method 1', method2_name='Method 2'):
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
        
        Args:
            scores1: Scores from method 1
            scores2: Scores from method 2
            method1_name: Name of method 1
            method2_name: Name of method 2
            
        Returns:
            dict: Test results
        """
        try:
            statistic, p_value = wilcoxon(scores1, scores2)
            
            result = {
                'method1': method1_name,
                'method2': method2_name,
                'method1_median': np.median(scores1),
                'method2_median': np.median(scores2),
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'better_method': method1_name if np.median(scores1) > np.median(scores2) else method2_name
            }
            
            self.test_results['wilcoxon'] = result
            
            print(f"\nWilcoxon Test: {method1_name} vs {method2_name}")
            print(f"  {method1_name} median: {result['method1_median']:.4f}")
            print(f"  {method2_name} median: {result['method2_median']:.4f}")
            print(f"  Statistic: {statistic:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if result['significant'] else 'No'}")
            
            return result
            
        except Exception as e:
            print(f"Wilcoxon test error: {str(e)}")
            return None
    
    def mann_whitney_test(self, scores1, scores2, method1_name='Method 1', method2_name='Method 2'):
        """
        Mann-Whitney U test (for independent samples)
        
        Args:
            scores1: Scores from method 1
            scores2: Scores from method 2
            method1_name: Name of method 1
            method2_name: Name of method 2
            
        Returns:
            dict: Test results
        """
        statistic, p_value = mannwhitneyu(scores1, scores2, alternative='two-sided')
        
        result = {
            'method1': method1_name,
            'method2': method2_name,
            'method1_median': np.median(scores1),
            'method2_median': np.median(scores2),
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        self.test_results['mann_whitney'] = result
        
        print(f"\nMann-Whitney U Test: {method1_name} vs {method2_name}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if result['significant'] else 'No'}")
        
        return result
    
    def friedman_test(self, *method_scores, method_names=None):
        """
        Friedman test (non-parametric alternative to repeated measures ANOVA)
        Compares multiple methods
        
        Args:
            *method_scores: Variable number of score arrays
            method_names: List of method names
            
        Returns:
            dict: Test results
        """
        if method_names is None:
            method_names = [f"Method {i+1}" for i in range(len(method_scores))]
        
        statistic, p_value = friedmanchisquare(*method_scores)
        
        result = {
            'methods': method_names,
            'n_methods': len(method_scores),
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        self.test_results['friedman'] = result
        
        print(f"\nFriedman Test (comparing {len(method_scores)} methods)")
        print(f"  Methods: {', '.join(method_names)}")
        print(f"  Statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if result['significant'] else 'No'}")
        
        return result
    
    def effect_size_cohens_d(self, scores1, scores2):
        """
        Calculate Cohen's d effect size
        
        Args:
            scores1: Scores from method 1
            scores2: Scores from method 2
            
        Returns:
            float: Cohen's d
        """
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
        n1, n2 = len(scores1), len(scores2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        
        # Interpretation
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        print(f"\nCohen's d Effect Size: {cohens_d:.4f} ({interpretation})")
        
        return cohens_d
    
    def confidence_interval(self, scores, confidence=0.95):
        """
        Calculate confidence interval
        
        Args:
            scores: Array of scores
            confidence: Confidence level (default: 0.95)
            
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        mean = np.mean(scores)
        std_err = stats.sem(scores)
        
        # Calculate confidence interval
        ci = stats.t.interval(
            confidence,
            len(scores) - 1,
            loc=mean,
            scale=std_err
        )
        
        print(f"\n{confidence*100}% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  Mean: {mean:.4f}")
        
        return ci
    
    def compare_methods_comprehensive(self, ga_scores, traditional_scores, 
                                     method_names=None):
        """
        Comprehensive comparison of GA vs traditional methods
        
        Args:
            ga_scores: GA scores
            traditional_scores: Dictionary of traditional method scores
            method_names: Names of traditional methods
            
        Returns:
            dict: Comprehensive comparison results
        """
        if method_names is None:
            method_names = list(traditional_scores.keys())
        
        print("\n" + "="*60)
        print("COMPREHENSIVE STATISTICAL COMPARISON")
        print("="*60)
        
        results = {}
        
        # Compare GA with each traditional method
        for method_name, scores in traditional_scores.items():
            print(f"\nComparing GA vs {method_name}")
            print("-"*60)
            
            # T-test
            t_result = self.paired_t_test(ga_scores, scores, 'GA', method_name)
            
            # Effect size
            cohens_d = self.effect_size_cohens_d(ga_scores, scores)
            
            results[method_name] = {
                't_test': t_result,
                'effect_size': cohens_d
            }
        
        # Friedman test (all methods together)
        if len(traditional_scores) > 1:
            print("\n" + "-"*60)
            all_scores = [ga_scores] + list(traditional_scores.values())
            all_names = ['GA'] + method_names
            friedman_result = self.friedman_test(*all_scores, method_names=all_names)
            results['friedman'] = friedman_result
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        
        return results
    
    def descriptive_statistics(self, scores, method_name='Method'):
        """
        Calculate descriptive statistics
        
        Args:
            scores: Array of scores
            method_name: Name of method
            
        Returns:
            dict: Descriptive statistics
        """
        stats_dict = {
            'method': method_name,
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores, ddof=1),
            'min': np.min(scores),
            'max': np.max(scores),
            'q1': np.percentile(scores, 25),
            'q3': np.percentile(scores, 75),
            'iqr': np.percentile(scores, 75) - np.percentile(scores, 25)
        }
        
        print(f"\nDescriptive Statistics for {method_name}:")
        print(f"  Mean:   {stats_dict['mean']:.4f}")
        print(f"  Median: {stats_dict['median']:.4f}")
        print(f"  Std:    {stats_dict['std']:.4f}")
        print(f"  Min:    {stats_dict['min']:.4f}")
        print(f"  Max:    {stats_dict['max']:.4f}")
        print(f"  Q1:     {stats_dict['q1']:.4f}")
        print(f"  Q3:     {stats_dict['q3']:.4f}")
        print(f"  IQR:    {stats_dict['iqr']:.4f}")
        
        return stats_dict


# Example usage
if __name__ == '__main__':
    print("="*60)
    print("STATISTICAL ANALYSIS EXAMPLE")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    
    ga_scores = np.random.normal(0.85, 0.05, 10)
    chi_square_scores = np.random.normal(0.80, 0.06, 10)
    mi_scores = np.random.normal(0.78, 0.07, 10)
    rf_scores = np.random.normal(0.82, 0.05, 10)
    
    # Initialize analyzer
    analyzer = StatisticalAnalysis()
    
    # Compare GA vs Chi-Square
    print("\n" + "="*60)
    print("GA vs CHI-SQUARE")
    print("="*60)
    
    t_result = analyzer.paired_t_test(ga_scores, chi_square_scores, 'GA', 'Chi-Square')
    cohens_d = analyzer.effect_size_cohens_d(ga_scores, chi_square_scores)
    
    # Comprehensive comparison
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON")
    print("="*60)
    
    traditional = {
        'Chi-Square': chi_square_scores,
        'Mutual Info': mi_scores,
        'RF Importance': rf_scores
    }
    
    results = analyzer.compare_methods_comprehensive(ga_scores, traditional)
    
    # Descriptive statistics
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    
    analyzer.descriptive_statistics(ga_scores, 'GA')
    analyzer.descriptive_statistics(chi_square_scores, 'Chi-Square')

