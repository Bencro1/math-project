import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("python-analysis/general_data.csv")
print(df['NumCompaniesWorked'].value_counts().sort_index())

# Statistiques descriptives de NumCompaniesWorked
print(df['NumCompaniesWorked'].describe())

# Visualisation de la distribution
plt.figure(figsize=(8,4))
sns.histplot(df['NumCompaniesWorked'], bins=20, kde=True)
plt.title('Distribution de NumCompaniesWorked')
plt.xlabel('Nombre de compagnies travaillées')
plt.ylabel('Fréquence')
plt.show()

# Valeurs manquantes
print("Valeurs manquantes :", df['NumCompaniesWorked'].isnull().sum())