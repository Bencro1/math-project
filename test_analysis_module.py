
import sys
import os
sys.path.insert(0, os.path.abspath("python-analysis"))

import analysis_module as am

def run_tests():
    loader = am.DataLoader()
    loader.load_csv_files()
    loader.load_excel_file()
    loader.merge_data()
    loader.clean_data()

    analysis = am.Analysis(loader.df)
    print("Discrete statistics:")
    import scipy.stats as stats
    cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome']
    discrete_stats = {}
    total_employees = len(analysis.df)
    for col in cols:
        if col == 'Attrition':
            attrition_yes_count = analysis.df[analysis.df['Attrition'] == 'Yes'].shape[0]
            discrete_stats[col] = {
                'count': attrition_yes_count,
                'percentage': (attrition_yes_count / total_employees) * 100
            }
        elif col == 'MaritalStatus':
            mode_val = analysis.df[col].mode()[0]
            discrete_stats[col] = {
                'mode': mode_val,
                'min': None,
                'max': None,
                'mean': None,
                'median': None,
                'range': None,
                'variance': None,
                'std_dev': None
            }
        else:
            col_data = analysis.df[col]
            discrete_stats[col] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'mode': col_data.mode()[0],
                'range': col_data.max() - col_data.min(),
                'variance': col_data.var(),
                'std_dev': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
    from tabulate import tabulate
    # Convert discrete_stats dict to a table format for pretty printing
    headers = ["Variable", "Mean", "Median", "Mode", "Range", "Variance", "Std Dev", "Min", "Max"]
    table = []
    for var, stats in discrete_stats.items():
        row = [
            var,
            f"{stats.get('mean', ''):.2f}" if stats.get('mean') is not None else "",
            f"{stats.get('median', ''):.2f}" if stats.get('median') is not None else "",
            f"{stats.get('mode', '')}" if stats.get('mode') is not None else "",
            f"{stats.get('range', ''):.2f}" if stats.get('range') is not None else "",
            f"{stats.get('variance', ''):.2f}" if stats.get('variance') is not None else "",
            f"{stats.get('std_dev', ''):.2f}" if stats.get('std_dev') is not None else "",
            f"{stats.get('min', ''):.2f}" if stats.get('min') is not None else "",
            f"{stats.get('max', ''):.2f}" if stats.get('max') is not None else ""
        ]
        table.append(row)
    print("\nDiscrete Statistics Table:")
    print(tabulate(table, headers=headers, tablefmt="psql"))

    print("Plotting attrition vs all variables...")
    analysis.plot_attrition_vs_all()

    # Removed plotting StandardHours distribution as method does not exist
    # print("Plotting StandardHours distribution...")
    # analysis.plot_standard_hours_distribution()

    # Removed job involvement vs attrition plot as per user request
    # print("Plotting job involvement vs attrition...")
    # analysis.plot_job_involvement_vs_attrition()

    # Removed number of companies worked vs attrition plot as per user request
    # print("Plotting number of companies worked vs attrition...")
    # analysis.plot_num_companies_vs_attrition()

    print("Plotting percent salary hike vs attrition...")
    analysis.plot_percent_salary_hike_vs_attrition()

    print("Plotting relationship satisfaction vs attrition...")
    analysis.plot_relationship_satisfaction_vs_attrition()

    print("Plotting years with current manager vs attrition...")
    analysis.plot_years_with_curr_manager_vs_attrition()

    print("Plotting years since last promotion vs attrition...")
    analysis.plot_years_since_last_promotion_vs_attrition()

    print("Plotting training times last year vs attrition...")
    analysis.plot_training_time_last_year_vs_attrition()

    print("Plotting total working years vs attrition...")
    analysis.plot_total_working_years_vs_attrition()

    print("Plotting stock option level vs attrition...")
    analysis.plot_stock_option_level_vs_attrition()

    print("Plotting percentage bar charts for selected variables...")
    analysis.plot_percentage_bars()

    print("Printing contingency tables...")
    analysis.contingency_tables()

    analysis.prepare_data_for_modeling()
    analysis.train_logistic_regression()

    cm, cr = analysis.evaluate_model()
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    coeffs = analysis.get_model_coefficients()
    print("Model Coefficients:")
    print(coeffs)

if __name__ == "__main__":
    run_tests()
