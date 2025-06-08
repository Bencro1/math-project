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
    print("Descriptive statistics:")
    print(analysis.descriptive_statistics())

    # Removed plotting distribution of EnvironmentSatisfaction
    # Removed plotting boxplot of Attrition vs MonthlyIncome
    # Removed plotting correlation matrix

    print("Plotting attrition vs all variables...")
    analysis.plot_attrition_vs_all()

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
