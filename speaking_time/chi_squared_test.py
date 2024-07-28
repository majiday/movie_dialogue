from scipy.stats import chi2_contingency

def perform_chi_squared_test(contingency_table):
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    return chi2, p, dof
