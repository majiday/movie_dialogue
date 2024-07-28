import data_processing as dp
import analysis
import visualization

def main():
    # Load and process data
    df = dp.load_data('utterances.jsonl')
    df = dp.filter_data(df)
    df = dp.extract_season(df)
    contingency_table = dp.create_contingency_table(df)

    # Statistical analysis
    chi2, p, dof = analysis.perform_chi_squared_test(contingency_table)

    # Visualization
    visualization.plot_bar_chart(contingency_table)
    visualization.plot_heatmap(contingency_table)

if __name__ == "__main__":
    main()
