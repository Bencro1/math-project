import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

class DataLoader:
    def __init__(self, data_dir="python-analysis/"):
        self.data_dir = data_dir
        self.employee = None
        self.manager = None
        self.general = None
        self.in_time = None
        self.out_time = None
        self.data_dictionary = None
        self.df = None

    def load_csv_files(self):
        self.employee = pd.read_csv(f"{self.data_dir}employee_survey_data.csv")
        self.manager = pd.read_csv(f"{self.data_dir}manager_survey_data.csv")
        self.general = pd.read_csv(f"{self.data_dir}general_data.csv")
        self.in_time = pd.read_csv(f"{self.data_dir}in_time.csv")
        self.out_time = pd.read_csv(f"{self.data_dir}out_time.csv")

    def load_excel_file(self):
        self.data_dictionary = pd.read_excel(f"{self.data_dir}data_dictionary.xlsx")

    def merge_data(self):
        if self.employee is None or self.manager is None or self.general is None:
            raise ValueError("CSV files not loaded yet. Call load_csv_files() first.")
        self.df = self.general.merge(self.employee, on='EmployeeID') \
                              .merge(self.manager, on='EmployeeID')

    def clean_data(self):
        if self.df is None:
            raise ValueError("Data not merged yet. Call merge_data() first.")
        self.df = self.df.dropna()

class Analysis:
    def __init__(self, df):
        self.df = df.copy()
        self.df_encoded = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def descriptive_statistics(self):
        return self.df.describe()

    def plot_attrition_vs_all(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        cols = ['Age', 'EducationField', 'JobRole', 'MonthlyIncome', 'YearsAtCompany']
        for col in cols:
            plt.figure(figsize=(10,6))
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # For numeric variables, create a histogram with hue=Attrition
                sns.histplot(data=self.df, x=col, hue='Attrition', multiple='dodge', shrink=0.8, bins=20)
                plt.title(f"Number of Employees by {col} and Attrition")
                plt.xlabel(col)
                plt.ylabel("Number of Employees")
            else:
                # For categorical variables, create a countplot with hue=Attrition
                if col == 'WorkLifeBalance':
                    continue
                sns.countplot(data=self.df, x=col, hue='Attrition')
                plt.title(f"Number of Employees by {col} and Attrition")
                plt.xlabel(col)
                plt.ylabel("Number of Employees")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    def contingency_tables(self):
        from tabulate import tabulate
        print("Contingency Tables for selected variables:")
        variables = ['BusinessTravel', 'Department', 'Gender', 'MaritalStatus']
        for var in variables:
            print(f"\nContingency Table for {var} and Attrition:")
            ct = pd.crosstab(self.df[var], self.df['Attrition'], margins=True)
            print(tabulate(ct, headers='keys', tablefmt='psql'))

    def plot_percentage_bars(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        variables = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
        for var in variables:
            plt.figure(figsize=(8,6))
            ct = pd.crosstab(self.df[var], self.df['Attrition'], normalize='index') * 100
            counts = self.df[var].value_counts().reindex([1, 2, 3, 4], fill_value=0)
            ax = ct.plot(kind='bar', stacked=True)
            plt.title(f"Percentage of Attrition by {var}")
            plt.xlabel(var)
            plt.ylabel("Percentage (%)")
            plt.legend(title='Attrition')
            plt.xticks(rotation=45, ha='right')
            # Add count labels on top of bars, only once at the top of each column
            labels = [str(counts.get(i, 0)) for i in range(1, 5)]
            for i, container in enumerate(ax.containers):
                if i == 0:
                    ax.bar_label(container, labels=labels)
                else:
                    ax.bar_label(container, labels=['']*len(labels))
            plt.tight_layout()
            plt.show()

    def plot_distribution(self, column, bins=5):
        plt.figure(figsize=(8,4))
        sns.histplot(self.df[column], bins=bins, kde=False)
        plt.title(f"Distribution of {column}")
        plt.show()

    def plot_boxplot(self, x_col, y_col):
        plt.figure(figsize=(8,4))
        sns.boxplot(x=x_col, y=y_col, data=self.df)
        plt.title(f"{y_col} by {x_col}")
        plt.show()

    def plot_correlation_matrix(self):
        plt.figure(figsize=(12,10))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    def prepare_data_for_modeling(self, target_col='Attrition'):
        # Encode target variable
        self.df[target_col] = self.df[target_col].map({'Yes':1, 'No':0})
        # Encode categorical variables except EmployeeID and target
        self.df_encoded = pd.get_dummies(self.df.drop(['EmployeeID'], axis=1), drop_first=True)
        X = self.df_encoded.drop(target_col, axis=1)
        y = self.df_encoded[target_col]
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

    def train_logistic_regression(self, max_iter=1000):
        self.model = LogisticRegression(max_iter=max_iter)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        cr = classification_report(self.y_test, y_pred)
        return cm, cr

    def get_model_coefficients(self):
        coefficients = pd.DataFrame({
            'Feature': self.df_encoded.drop('Attrition', axis=1).columns,
            'Coefficient': self.model.coef_[0]
        }).sort_values(by='Coefficient', ascending=False)
        return coefficients

    def plot_feature_importance(self, top_n=20):
        coeffs = self.get_model_coefficients()
        plt.figure(figsize=(10,6))
        sns.barplot(x='Coefficient', y='Feature', data=coeffs.head(top_n))
        plt.title(f"Top {top_n} Features Influencing Attrition")
        plt.tight_layout()
        plt.show()

    def summary_report(self, top_n=10):
        coeffs = self.get_model_coefficients()
        print("Summary Report: Key Factors Influencing Employee Attrition")
        print("-" * 60)
        print("Top positive factors (increase likelihood of attrition):")
        print(coeffs.head(top_n))
        print("\nTop negative factors (decrease likelihood of attrition):")
        print(coeffs.tail(top_n).sort_values(by='Coefficient'))
        print("-" * 60)

    def plot_attrition_comparison(self, column):
        plt.figure(figsize=(8,4))
        sns.boxplot(x='Attrition', y=column, data=self.df)
        plt.title(f"{column} by Attrition")
        plt.show()

    def plot_job_involvement_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['JobInvolvement'], self.df['Attrition'], normalize='index') * 100
        counts = self.df['JobInvolvement'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=True)
        plt.title("Percentage of Attrition by Job Involvement")
        plt.xlabel("Job Involvement")
        plt.ylabel("Percentage (%)")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        # Add count labels on top of bars, only once at the top of each column
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()

    def plot_num_companies_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['NumCompaniesWorked'], self.df['Attrition'])
        counts = self.df['NumCompaniesWorked'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=False)
        plt.title("Number of Employees by NumCompaniesWorked and Attrition")
        plt.xlabel("Number of Companies Worked")
        plt.ylabel("Number of Employees")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        # Add count labels on top of bars, only once at the top of each column
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()

    def plot_percent_salary_hike_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['PercentSalaryHike'], self.df['Attrition'], normalize='index') * 100
        counts = self.df['PercentSalaryHike'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=True)
        plt.title("Percentage of Attrition by Percent Salary Hike")
        plt.xlabel("Percent Salary Hike")
        plt.ylabel("Percentage (%)")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        # Add count labels on top of bars, only once at the top of each column
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()

    def plot_relationship_satisfaction_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        if 'RelationshipSatisfaction' not in self.df.columns:
            print("Column 'RelationshipSatisfaction' not found in data. Skipping plot.")
            return

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['RelationshipSatisfaction'], self.df['Attrition'], normalize='index') * 100
        counts = self.df['RelationshipSatisfaction'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=True)
        plt.title("Percentage of Attrition by Relationship Satisfaction")
        plt.xlabel("Relationship Satisfaction")
        plt.ylabel("Percentage (%)")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        # Add count labels on top of bars, only once at the top of each column
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()

    def plot_years_with_curr_manager_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        if 'YearsWithCurrManager' not in self.df.columns:
            print("Column 'YearsWithCurrManager' not found in data. Skipping plot.")
            return

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['YearsWithCurrManager'], self.df['Attrition'], normalize='index') * 100
        counts = self.df['YearsWithCurrManager'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=True)
        plt.title("Percentage of Attrition by Years With Current Manager")
        plt.xlabel("Years With Current Manager")
        plt.ylabel("Percentage (%)")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()

    def plot_years_since_last_promotion_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        if 'YearsSinceLastPromotion' not in self.df.columns:
            print("Column 'YearsSinceLastPromotion' not found in data. Skipping plot.")
            return

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['YearsSinceLastPromotion'], self.df['Attrition'], normalize='index') * 100
        counts = self.df['YearsSinceLastPromotion'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=True)
        plt.title("Percentage of Attrition by Years Since Last Promotion")
        plt.xlabel("Years Since Last Promotion")
        plt.ylabel("Percentage (%)")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()

    def plot_training_time_last_year_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        if 'TrainingTimesLastYear' not in self.df.columns:
            print("Column 'TrainingTimesLastYear' not found in data. Skipping plot.")
            return

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['TrainingTimesLastYear'], self.df['Attrition'], normalize='index') * 100
        counts = self.df['TrainingTimesLastYear'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=True)
        plt.title("Percentage of Attrition by Training Times Last Year")
        plt.xlabel("Training Times Last Year")
        plt.ylabel("Percentage (%)")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()

    def plot_total_working_years_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        if 'TotalWorkingYears' not in self.df.columns:
            print("Column 'TotalWorkingYears' not found in data. Skipping plot.")
            return

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['TotalWorkingYears'], self.df['Attrition'], normalize='index') * 100
        counts = self.df['TotalWorkingYears'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=True)
        plt.title("Percentage of Attrition by Total Working Years")
        plt.xlabel("Total Working Years")
        plt.ylabel("Percentage (%)")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()

    def plot_stock_option_level_vs_attrition(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        if 'StockOptionLevel' not in self.df.columns:
            print("Column 'StockOptionLevel' not found in data. Skipping plot.")
            return

        plt.figure(figsize=(8,6))
        ct = pd.crosstab(self.df['StockOptionLevel'], self.df['Attrition'], normalize='index') * 100
        counts = self.df['StockOptionLevel'].value_counts().sort_index()
        ax = ct.plot(kind='bar', stacked=True)
        plt.title("Percentage of Attrition by Stock Option Level")
        plt.xlabel("Stock Option Level")
        plt.ylabel("Percentage (%)")
        plt.legend(title='Attrition')
        plt.xticks(rotation=0)
        labels = [str(counts.get(i, 0)) for i in sorted(counts.index)]
        for i, container in enumerate(ax.containers):
            if i == 0:
                ax.bar_label(container, labels=labels)
            else:
                ax.bar_label(container, labels=['']*len(labels))
        plt.tight_layout()
        plt.show()
