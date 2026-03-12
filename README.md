# A/B Test Analysis for Article Thumbnails

In this project, we explore a dataset of user interactions with article thumbnails to understand how visual changes affect click behavior.  

The goal is to determine whether updating a thumbnail increases engagement, measured by clicks on the **“read more”** button, while monitoring key guardrails such as time on page, scroll depth, exit rate, and other interactions.

The `ABTest` Python class centralizes the workflow:

- **Data management:** load, append, and inspect experimental batches.
- **Balance checks:** confirm covariates are evenly distributed across control and treatment groups.
- **Power analysis:** estimate required sample size for a minimum detectable effect (MDE) and assess whether the current experiment is underpowered.
- **Statistical testing:** perform one-sided or two-sided Welch t-tests for binary KPIs.
- **Lift estimation:** compute absolute lift and confidence intervals.
- **Regression checks:** fit robust OLS regressions to validate results and account for batch effects.

**Directory structure:**
- `data/` — contains raw experiment CSVs.  

**Usage:**

1. Load your dataset into a pandas DataFrame.  
2. Initialize the `ABTest` class with the DataFrame and the name of your KPI.  
3. Run `power_analysis()` to check sample size requirements.  
4. Use `ab_ttest()` or `welch_t_test()` to evaluate the effect of your variant.  
5. Inspect lift with `lift_ci()` and optionally check robustness via `regression_rob_check()`.  

This project provides a framework for running, analyzing, and interpreting A/B tests for binary engagement metrics in a consistent, reproducible manner.