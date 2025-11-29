#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd  # pour manipuler les fichiers CSV
import io            # pour lire le fichier depuis la mémoire
from google.colab import files  # pour uploader le fichier depuis ton PC


# In[ ]:


uploaded = files.upload()  # ouvre un bouton pour choisir le fichier


# In[ ]:


file_name = list(uploaded.keys())[0]
print("Nom exact du fichier :", file_name)


# In[ ]:


df = pd.read_csv(io.BytesIO(uploaded[file_name]), encoding='latin1')


# In[ ]:


df.head()  # affiche les 5 premières lignes du tableau


# In[ ]:


import pandas as pd
import os

possible = ['DataCoSupplyChainDataset.csv', '/content/DataCoSupplyChainDataset.csv', '/mnt/data/DataCoSupplyChainDataset.csv']
df = None
for p in possible:
    if os.path.exists(p):
        df = pd.read_csv(p, encoding='latin-1')
        print("Chargé depuis:", p)
        break

if df is None:
    raise FileNotFoundError("DataCoSupplyChainDataset.csv introuvable. Refaire l'étape 1b.")

print("Shape:", df.shape)
df.head()


# In[ ]:


import numpy as np

# création targets (suivant consignes)
df['fraud'] = np.where(df['Order Status'] == 'SUSPECTED_FRAUD', 1, 0)
df['late_delivery'] = np.where(df['Delivery Status'] == 'Late delivery', 1, 0)

print("fraud distribution:")
print(df['fraud'].value_counts(normalize=True))
print("\nlate_delivery distribution:")
print(df['late_delivery'].value_counts(normalize=True))


# In[ ]:


cols_to_drop = [
    'Product Image', 'Product Name', 'shipping date (DateOrders)',
    'Order Date (DateOrders)', 'Customer Email', 'Customer Full Name',
    'Customer Password'
]
cols_present = [c for c in cols_to_drop if c in df.columns]
print("Colonnes supprimées:", cols_present)
df = df.drop(columns=cols_present)
print("Nouvelle shape:", df.shape)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Nombre colonnes object à encoder:", len(cat_cols))
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print("Encodage terminé.")


# In[ ]:


features = [c for c in df.columns if c not in ['fraud','late_delivery']]
X = df[features].copy()
X_fraud, y_fraud = X.copy(), df['fraud'].copy()
X_late,  y_late  = X.copy(), df['late_delivery'].copy()
print("Nombre features:", X.shape[1])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.impute import SimpleImputer # Import SimpleImputer
import pandas as pd # Import pandas for DataFrame operations

def run_models(X, y, test_size=0.2):
    # Drop columns that are entirely NaN before imputation
    X_cleaned = X.dropna(axis=1, how='all')
    print(f"Dropped {len(X.columns) - len(X_cleaned.columns)} columns that were entirely NaN.")

    # Impute remaining missing values
    imputer = SimpleImputer(strategy='mean') # Using mean imputation as a general strategy
    X_imputed = imputer.fit_transform(X_cleaned)
    X_imputed = pd.DataFrame(X_imputed, columns=X_cleaned.columns) # Convert back to DataFrame for column names

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=test_size, stratify=y, random_state=42)
    models = {
        'Logistic': LogisticRegression(max_iter=500),
        'Lasso(L1)': LogisticRegression(penalty='l1', solver='liblinear', max_iter=500),
        'RF': RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    }
    res = {}
    for name, m in models.items():
        print("Training", name)
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        res[name] = {
            'accuracy': accuracy_score(y_test, p),
            'recall': recall_score(y_test, p, zero_division=0),
            'f1': f1_score(y_test, p, zero_division=0)
        }
    return res

print("Fraud:")
res_f = run_models(X_fraud, y_fraud, test_size=0.2)
print(res_f)
print("\nLate delivery:")
res_l = run_models(X_late, y_late, test_size=0.2)
print(res_l)


# In[ ]:


# Réentraîner un RF complet (n_estimators élevé) pour extraire importances
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)

# Ex: pour fraud en 20% test
Xtr, Xte, ytr, yte = train_test_split(X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42)
rf.fit(Xtr, ytr)
importances = pd.Series(rf.feature_importances_, index=X_fraud.columns).sort_values(ascending=False)
print("Top 5 features fraud:")
display(importances.head(5))

# Idem pour late_delivery
Xtr, Xte, ytr, yte = train_test_split(X_late, y_late, test_size=0.2, stratify=y_late, random_state=42)
rf2 = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
rf2.fit(Xtr, ytr)
importances2 = pd.Series(rf2.feature_importances_, index=X_late.columns).sort_values(ascending=False)
print("Top 5 features late_delivery:")
display(importances2.head(5))


# In[ ]:


# Vérifie où tu te trouves et liste les fichiers
get_ipython().system('pwd')
get_ipython().system('ls -la')
get_ipython().system('ls -la /content')
get_ipython().system('ls -la /mnt/data')


# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt


# ## Dataset Summary

# In[ ]:


print("Shape of the DataFrame:", df.shape)
print("\nDataFrame Info:")
df.info()


# In[ ]:


print("\nMissing values per column:")
display(df.isnull().sum()[df.isnull().sum() > 0])


# ## Identify Leakage in Late Delivery Prediction

# In[ ]:


print("Column used to create 'late_delivery' target: 'Delivery Status' and 'Late_delivery_risk'")
print("'Delivery Status' is present in X_late features: ", 'Delivery Status' in X_late.columns)
print("'Late_delivery_risk' is present in X_late features: ", 'Late_delivery_risk' in X_late.columns)


# ## Identify Leakage in Fraud Prediction

# In[ ]:


print("Column used to create 'fraud' target: 'Order Status'")
print("'Order Status' is present in X_fraud features: ", 'Order Status' in X_fraud.columns)


# ## Propose Feature Removal

# Based on the analysis, we propose removing the following features to mitigate data leakage:
# 
# *   From `X_fraud`: `'Order Status'`
# *   From `X_late`: `'Delivery Status'`, `'Late_delivery_risk'`

# In[ ]:


get_ipython().system('ls /content/results')


# In[ ]:





# In[ ]:


from google.colab import files
files.download('/content/results/models_summary.csv')


# In[ ]:


get_ipython().system('ls /content/results')


# In[ ]:


get_ipython().system('ls /content')
get_ipython().system('ls /content/drive/MyDrive')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('ls /content/drive/MyDrive')


# In[ ]:


get_ipython().system('mkdir -p /content/results')


# In[ ]:


get_ipython().system('find /content/drive/MyDrive -name "DataCoSupplyChainDataset.csv"')


# In[ ]:


get_ipython().system('find /content/drive/MyDrive -name "DataCoSupplyChainDataset.csv"')


# In[ ]:


get_ipython().system('ls /content/drive')


# In[ ]:


get_ipython().system('ls /content/drive/MyDrive')


# In[ ]:


get_ipython().system('ls /content')


# In[ ]:


from google.colab import files
files.download('/content/DataCoSupplyChainDataset.csv')


# In[ ]:


get_ipython().system('cd /content && zip -r results.zip results')


# In[ ]:


get_ipython().system('ls -R /content')


# In[ ]:


import os

os.makedirs('/content/results', exist_ok=True)
print("Dossier results créé.")


# In[ ]:


get_ipython().system('ls /content')


# In[ ]:


plt.savefig('/content/results/plot1.png')


# In[ ]:


import pandas as pd

df = pd.read_csv('/content/DataCoSupplyChainDataset.csv')
df.head()


# In[ ]:


df = pd.read_csv('/content/DataCoSupplyChainDataset.csv', encoding='latin1')
df.head()


# In[ ]:


df = pd.read_csv('/content/DataCoSupplyChainDataset.csv', encoding='latin1')
df.head()


# In[ ]:


summary = df.describe()
summary.to_csv('/content/results/summary.csv')
print("Résumé enregistré.")


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
df['Days for shipping (real)'].hist()
plt.title("Histogram - Days for Shipping (Real)")
plt.xlabel("Days")
plt.ylabel("Count")
plt.savefig('/content/results/hist_shipping.png')
plt.close()

print("Graphique enregistré.")


# In[ ]:


get_ipython().system('ls /content/results')


# In[ ]:


from google.colab import files
files.download('/content/results/summary.csv')
files.download('/content/results/hist_shipping.png')


# In[75]:


((racine))
│
├── Assignment2_Rassim.ipynb   (obligatoire)
├── pipeline.py                 (obligatoire)
├── Assignment2_Report_Rassim.pdf  (obligatoire)
├── DataCoSupplyChainDataset.csv   (facultatif)
├── README.md                   (recommandé)
│
└── results/                    (obligatoire)
       ├── summary.csv
       ├── hist_shipping.png
       ├── plot1.png
       └── autres fichiers générés…


# In[76]:


from google.colab import files
files.download('/content/results/summary.csv')
files.download('/content/results/hist_shipping.png')
files.download('/content/results/plot1.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Task
# Fix the syntax error by replacing the non-printable character U+00A0 with a regular space in the `import matplotlib.pyplot as plt` line in cell `mCTsXUr8-4IF`, then re-execute the cell.

# ## Fix Syntax Error
# 
# ### Subtask:
# Replace the non-printable character U+00A0 (non-breaking space) with a regular space in the line `import matplotlib.pyplot as plt` within cell `mCTsXUr8-4IF`.
# 

# **Reasoning**:
# The previous code block in cell `mCTsXUr8-4IF` contains a syntax error due to a non-printable character. I will replace the content of this cell with the corrected code, ensuring the non-breaking space is replaced by a regular space.
# 
# 

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt


# ## Re-execute Cell
# 
# ### Subtask:
# Execute the corrected cell `mCTsXUr8-4IF` to ensure the syntax error is resolved and all necessary libraries are imported successfully.
# 

# ## Summary:
# 
# ### Data Analysis Key Findings
# *   A `SyntaxError` was identified in the `import matplotlib.pyplot as plt` line within cell `mCTsXUr8-4IF`, caused by a non-printable character (U+00A0).
# *   The error was successfully resolved by replacing the non-printable character with a regular space.
# *   The cell was re-executed, ensuring all necessary libraries were imported without syntax errors.
# 
# ### Insights or Next Steps
# *   The successful re-execution of the cell confirms that all required libraries are now correctly imported and available for subsequent analysis.
# 
