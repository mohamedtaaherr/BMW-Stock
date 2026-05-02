# ===================== IMPORTS =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score,
    davies_bouldin_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
# ===================== 1. DATA UNDERSTANDING =====================
df = pd.read_csv(r"D:\Dataset\BMW\bmw_advanced_features.csv")

print("\n=== RANDOM SAMPLE ===")
print(df.sample(5))

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== INFO ===")
print(df.info())

print("\n=== COLUMNS ===")
print(df.columns)

print("\n=== DESCRIPTION ===")
print(df.describe())

print("\n=== UNIQUE VALUES ===")
print(df.nunique())

# ===================== 1.2 DATA CLEAN =====================
df.drop_duplicates(inplace=True)

# Fix column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Convert date if exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Missing values
df.fillna(method='ffill', inplace=True)

# Remove invalid values (example)
df = df[df['close'] > 0]

# ===================== OUTLIER FLAGGING (IQR) =====================
def detect_outliers(df, cols):
    outliers = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR))
    return outliers

numeric_cols = df.select_dtypes(include=np.number).columns
outliers = detect_outliers(df, numeric_cols)

# ===================== 1.3 EDA =====================
# Univariate
df[numeric_cols].hist(figsize=(12,10))
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scatter
sns.scatterplot(x='volume', y='close', data=df)
plt.show()

# ===================== MULTICOLLINEARITY (VIF) =====================
X_vif = df[numeric_cols].drop(columns=['close'], errors='ignore')
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i)
                   for i in range(len(X_vif.columns))]
print("\n=== VIF ===")
print(vif_data.sort_values(by="VIF", ascending=False).head(10))

# ===================== 1.4 FEATURE ENGINEERING =====================
#   Predict FUTURE
df['target_reg'] = df['close'].shift(-1)

# Classification target (UP or DOWN)
df['target_cls'] = (df['target_reg'] > df['close']).astype(int)

df.dropna(inplace=True)

# Remove leakage columns
leak_cols = ['close', 'adj_close', 'target_reg']
X = df.drop(columns=[col for col in leak_cols if col in df.columns] + ['target_cls'])

y_cls = df['target_cls']
y_reg = df['target_reg']

# ===================== SCALING =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===================== TRAIN TEST SPLIT (TIME SERIES SAFE) =====================
split = int(len(X)*0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_cls_train, y_cls_test = y_cls[:split], y_cls[split:]
y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]

# ===================== 2. CLUSTERING =====================
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

print("\n=== CLUSTERING METRICS ===")
print("Silhouette Score:", silhouette_score(X_scaled, clusters))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, clusters))

# ===================== 3. CLASSIFICATION =====================
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_cls_train)

y_pred_cls = clf.predict(X_test)

print("\n=== CLASSIFICATION RESULTS ===")
print("Accuracy:", accuracy_score(y_cls_test, y_pred_cls))
print("F1 Score:", f1_score(y_cls_test, y_pred_cls))
print("ROC-AUC:", roc_auc_score(y_cls_test, clf.predict_proba(X_test)[:,1]))
print("\nClassification Report:\n", classification_report(y_cls_test, y_pred_cls))

# ===================== OPTIONAL: EXTRA MODEL =====================
log_model = LogisticRegression()
log_model.fit(X_train, y_cls_train)

print("\nLogistic Accuracy:",
      accuracy_score(y_cls_test, log_model.predict(X_test)))

# ===================== REGRESSION =====================
reg = LinearRegression()
reg.fit(X_train, y_reg_train)

y_pred_reg = reg.predict(X_test)

print("\n=== REGRESSION RESULTS ===")
print("MAE:", mean_absolute_error(y_reg_test, y_pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_reg_test, y_pred_reg)))
print("R2:", r2_score(y_reg_test, y_pred_reg))

# ===================== PCA  =====================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=clusters)
plt.title("3D PCA Clusters")
plt.show()

print("\n FULL PIPELINE COMPLETED SUCCESSFULLY")