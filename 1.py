# Import necessary libraries
import pandas as pd 
import missingno as msno 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load the dataset (Assume the dataset is in CSV format)
df = pd.read_csv('ALLFLOWMETER_HIKARI2021.csv')

# Group by 'traffic_category' and count the occurrences
traffic_counts = df.groupby('traffic_category').size().reset_index(name='count')

# Plot the count of each traffic category using Plotly
fig = px.bar(traffic_counts, x='traffic_category', y='count', 
             title='Count of Traffic Category', 
             labels={'traffic_category': 'Traffic Category', 'count': 'Count'},
             text='count')

# Show the plot
fig.show()

# Import necessary libraries
import plotly.express as px
import plotly.graph_objects as go

# a) Visualize distributions of key features
# Visualize distribution of 'flow_pkts_per_sec'
fig = px.histogram(df, x='flow_pkts_per_sec', nbins=50, title='Distribution of flow_pkts_per_sec',
                   labels={'flow_pkts_per_sec': 'Flow Packets per Second'})
fig.show()

# Visualize distribution of 'payload_bytes_per_second'
fig = px.histogram(df, x='payload_bytes_per_second', nbins=50, title='Distribution of Payload Bytes per Second',
                   labels={'payload_bytes_per_second': 'Payload Bytes per Second'})
fig.show()

# Visualize distribution of flag counts (e.g., flow_FIN_flag_count)
flag_columns = ['flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'flow_ACK_flag_count']

# Creating subplots for each flag
fig = go.Figure()

for flag in flag_columns:
    fig.add_trace(go.Histogram(x=df[flag], name=flag, nbinsx=20))

fig.update_layout(title_text='Distribution of Flag Counts', barmode='overlay', xaxis_title='Flag Count', yaxis_title='Count')
fig.show()

# b) Analyze the balance of classes in the 'Label' column
label_counts = df['Label'].value_counts().reset_index()
label_counts.columns = ['Label', 'count']

fig = px.bar(label_counts, x='Label', y='count', title='Class Balance in Label Column',
             labels={'Label': 'Class Label', 'count': 'Count'}, text='count')
fig.show()

# c) Look for patterns in features like 'flow_pkts_per_sec', 'payload_bytes_per_second', and flag counts

# Scatter plot for flow_pkts_per_sec vs payload_bytes_per_second colored by Label
fig = px.scatter(df, x='flow_pkts_per_sec', y='payload_bytes_per_second', color='Label',
                 title='Flow Packets per Second vs Payload Bytes per Second',
                 labels={'flow_pkts_per_sec': 'Flow Packets per Second', 'payload_bytes_per_second': 'Payload Bytes per Second'})
fig.show()

# Correlation heatmap for flag counts and the two key features
key_features = ['flow_pkts_per_sec', 'payload_bytes_per_second'] + flag_columns
corr_matrix = df[key_features].corr()

fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap of Key Features and Flag Counts")
fig.show()





# Loading the dataset 
  
# Visualize missing values as a matrix 
msno.matrix(df) 
plt.show()
# a) Checking for missing values

missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)
# Import plotly for visualization
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

# a) Missing values visualization
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    fig = px.bar(missing_values, title="Missing Values per Column", labels={'index': 'Columns', 'value': 'Missing Count'})
    fig.show()
else:
    print("No missing values to visualize.")


'''
# b) Handling missing values (Imputation)
# For numerical columns, let's use the mean strategy for imputation.
num_cols = df.select_dtypes(include=[np.number]).columns
imputer_num = SimpleImputer(strategy='mean')

# For categorical columns, we'll use the most frequent strategy for imputation.
cat_cols = df.select_dtypes(include=[object]).columns
imputer_cat = SimpleImputer(strategy='most_frequent')

# Apply imputations
df[num_cols] = imputer_num.fit_transform(df[num_cols])
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Check again if all missing values are handled
print("Missing values after imputation:\n", df.isnull().sum())
traffic_category

'''
# d) Outlier Detection using Isolation Forest
# Dropping 'Label' column as it's the target
X = df.drop(columns=['Label','originh','responh','traffic_category'])

# Using Isolation Forest to detect outliers
iso_forest = IsolationForest(contamination=0.01, random_state=42)
outliers = iso_forest.fit_predict(X)

# Marking rows identified as outliers (-1) and removing them
df_cleaned = df[outliers != -1]

# Split data into features (X) and labels (y)
X_cleaned = df_cleaned.drop(columns=['Label'])
y_cleaned = df_cleaned['Label']

# Optionally, split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Display the cleaned dataset
print(f"Data shape after outlier removal: {df_cleaned.shape}")

#
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Fit Isolation Forest and get outliers
iso_forest = IsolationForest(contamination=0.01, random_state=42)
outliers = iso_forest.fit_predict(X_pca)

# Prepare data for plotting
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['outliers'] = outliers
df_pca['outliers'] = df_pca['outliers'].map({1: 'Inlier', -1: 'Outlier'})

# Scatter plot for inliers and outliers
fig = px.scatter(df_pca, x='PC1', y='PC2', color='outliers', title="Outlier Detection via Isolation Forest (PCA)",
                 labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
fig.show()

# Optional: Plot cumulative explained variance of PCA (if you want to explain the variance captured by PCs)
pca_explained_variance = pca.explained_variance_ratio_.cumsum()
fig = go.Figure(go.Scatter(x=np.arange(1, len(pca_explained_variance) + 1), 
                           y=pca_explained_variance, mode='lines+markers'))
fig.update_layout(title='Cumulative Explained Variance by PCA Components', 
                  xaxis_title='Number of Principal Components', 
                  yaxis_title='Cumulative Explained Variance')
fig.show()

