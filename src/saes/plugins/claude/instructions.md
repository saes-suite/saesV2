# Claude Instructions for SAES (Stochastic Algorithm Evaluation Suite)

When interacting with scientific benchmark datasets in SAES format:

1. **Verify CSV Schema First**:
   - Required columns: `algorithm`, `instance`, `metric`, `run_id`, `value`, `maximize`.
   - Ensure `run_id` is an integer seed and `maximize` is boolean.

2. **Statistical Methodology**:
   - For 2 algorithms: Check normality with Shapiro-Wilk and variance homogeneity with Levene's test. If variances are unequal, use Welch's t-test.
   - For >2 algorithms: Check assumptions. If homoscedasticity fails, use Welch's One-Way ANOVA.
   - Always report effect size (Cohen's d or Eta-squared) alongside p-values.
   - Clarify the difference between statistical significance ($p < \alpha$) and practical significance.

3. **Using SAES Tools**:
   - Use `validate_benchmark_csv` before running tests.
   - Use `run_statistical_analysis` to get comprehensive reports per `(instance, metric)` cell.
