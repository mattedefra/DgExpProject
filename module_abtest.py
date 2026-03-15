from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import statsmodels.formula.api as smf
from scipy import stats
import numpy as np
import pandas as pd

class ABTest:
    def __init__(self, df, **kwargs):
        """
        ABTest class for binary KPIs (clicked or not).

        Parameters
        ----------
        df : pd.DataFrame
            The dataset containing the experiment results.
        kpi : str, optional
            Column name for the binary KPI (default is 'kpy_y which is a clicked or not type of variable').
        """
        self.iteration = 1
        self.df = df
        self.df.rename(columns={'kpi_y':'clicked'}, inplace=True)
        self.kpi = kwargs.get('kpi', 'clicked')
        self.group_var = kwargs.get('groupvar', 'arm')

    def append_data(self, new_df):
        """
        Append new batch of data to existing dataframe.
        The original plan was to receive new batches of data periodically, but alas we ended up using simulated data
        and this method will not be used.
        """
        self.iteration += 1
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def balance(self, feature: str):
        """
        Show distribution of a feature across groups.
        """
        balance_table = pd.crosstab(self.df['arm'], self.df[feature], normalize='index')
        print(balance_table)

    def are_we_underpowered(self, actual, required):
        difference = actual - required
        return 'underpowered' if difference < 0 else 'appropriately powered'

    def power_analysis(self, **kwargs):
        """
        Estimate whether the current sample size is sufficient for detecting a minimum detectable effect (MDE).

        Default mde, alpha, power are respectively 0.02, 0.05, 0.80
        """
        baseline_rate = self.df[self.kpi].mean()
        mde_absolute = kwargs.get('mde', 0.02)
        expected_treatment = baseline_rate + mde_absolute
        alpha = kwargs.get('alpha', 0.05)
        power = kwargs.get('power', 0.80)

        effect_size = proportion_effectsize(baseline_rate, expected_treatment)

        analysis = NormalIndPower()
        required_n = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=1
        )

        actual_n = self.df[self.group_var].value_counts().min()
        print(
            f"Required sample size per group: {np.ceil(required_n)}\n"
            f"Actual sample size per group: {np.ceil(actual_n)}\n"
            f"Experiment is {self.are_we_underpowered(actual_n, required_n)}"
        )

    def ab_ttest(self, doprint=True, two_sided=False):
        """
        Perform one-sided Welch's t-test on a binary KPI between control and treatment.
        Always tests H1: treatment =/= control.
        Prints t-statistic and p-value.
        """
        control = self.df[self.df[self.group_var] == 'control'][self.kpi].dropna()
        treatment = self.df[self.df[self.group_var] == 'treatment'][self.kpi].dropna()

        #Welch t-test statistic
        t_stat, _ = stats.ttest_ind(treatment, control, equal_var=False)

        df = len(control) + len(treatment) - 2

        if two_sided:
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            p_value = 1 - stats.t.cdf(t_stat, df)

        if doprint:
            print("t-stat:", t_stat)
            print("p-value:", p_value)


    def welch_t_test(self, doprint=False):
        """
        Perform one-sided Welch's t-test on a binary KPI between control and treatment.
        Always tests H1: treatment > control.
        Returns treatment and control series, and optionally prints t-statistic and p-value.
        """
        control = self.df[self.df[self.group_var] == 'control'][self.kpi].dropna()
        treatment = self.df[self.df[self.group_var] == 'treatment'][self.kpi].dropna()

        # Safeguard: need at least 2 observations per group
        if len(control) < 2 or len(treatment) < 2:
            raise ValueError("Not enough observations in one of the groups to perform t-test.")

        # Safeguard: variance > 0
        if control.var(ddof=1) == 0 or treatment.var(ddof=1) == 0:
            raise ValueError("Zero variance in one of the groups; t-test cannot be computed.")

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(
            treatment,
            control,
            equal_var=False
        )

        # Always one-sided: H1 = treatment > control
        if t_stat > 0:
            p_value = p_value / 2
        else:
            p_value = 1 - (p_value / 2)

        if doprint:
            print("t-stat:", t_stat)
            print("one-sided p-value:", p_value)

        else: return treatment, control

    def lift_ci(self, one_sided=True):
        """
        Compute absolute lift and 95% confidence interval.
        If one_sided=True, compute CI for H1: treatment > control
        """
        treatment, control = self.welch_t_test()
        diff = treatment.mean() - control.mean()

        se = np.sqrt(
            treatment.var(ddof=1)/len(treatment) +
            control.var(ddof=1)/len(control)
        )

        if one_sided:
            z = 1.645  # 95% one-sided
            ci_low = diff - z * se
            ci_high = np.inf
        else:
            z = 1.96  # 95% two-sided
            ci_low = diff - z * se
            ci_high = diff + z * se

        print(f"Lift: {round(diff*100, 2)}%")
        if one_sided:
            print(f"One-sided 95% CI: [{ci_low:.4f}, ∞]")
        else:
            print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    def regression_rob_check(self):
        """
        Fit a robust OLS regression for the KPI on group and other covariates.
        """
        formula = f"{self.kpi} ~ C({self.group_var})"
        if self.iteration > 1:
            formula += f" + C(Data_Batch)"

        model = smf.ols(formula=formula, data=self.df).fit(cov_type='HC3')
        print(model.summary())