# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import the required libraries
(numpy, pandas, matplotlib, sklearn).

Step 3: Load the customer dataset containing attributes such as:

Age

Annual income

Spending score

Purchase frequency

Step 4: Select the relevant features for clustering
(Independent variables only; no target variable).

Step 5: Perform data preprocessing:

Handle missing values

Normalize or standardize the data

Step 6: Choose the number of clusters K
(using the Elbow Method if required).

Step 7: Initialize the K-Means clustering model with the selected value of K.

Step 8: Fit the K-Means model to the dataset.

Step 9: Assign each customer to a cluster based on the nearest centroid.

Step 10: Obtain the cluster labels for all customers.

Step 11: Visualize the customer segments using a scatter plot (optional).

Step 12: Analyze and interpret the formed customer clusters.

Step 13: Stop the program.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: 
RegisterNumber:  
*/
```
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Gender': ['Male','Female','Female','Male','Female','Male','Male','Female','Female','Male'],
    'Age': [19,21,20,23,31,22,35,30,25,28],
    'Annual Income (k$)': [15,16,17,18,19,20,21,22,23,24],
    'Spending Score (1-100)': [39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Select features for clustering
# ------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ------------------------------
# Step 3: Apply K-Means (choose clusters, e.g., 3)
# ------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)  # Automatically fits and assigns clusters

# ------------------------------
# Step 4: Visualize clusters
# ------------------------------
plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(X[df['Cluster']==i]['Annual Income (k$)'],
                X[df['Cluster']==i]['Spending Score (1-100)'],
                label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='yellow', label='Centroids', marker='X')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# ------------------------------
# Step 5: Show dataset with clusters
# ------------------------------
print(df)

## Output:
![K Means Clustering for Customer Segmentation](sam.png)

<img width="852" height="680" alt="image" src="https://github.com/user-attachments/assets/1bbb898e-2e87-4261-9f7d-5dd76b0cdd78" />
<img width="756" height="553" alt="image" src="https://github.com/user-attachments/assets/0dc9a02b-a35e-4804-a9f7-f55200132546" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
