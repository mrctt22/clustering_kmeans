# Modules
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns
from sklearn.datasets import (make_blobs, make_circles, make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

#%matplotlib inline
sns.set_context('notebook')
plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')

# Import the data
df = pd.read_csv(r".\data.csv",sep=";") 
print(df)

#df.plot(x='Stagione', y='Affitti')
#plt.show()

# Plot the data
#plt.scatter(df.iloc[:, 12], df.iloc[:, 13])
#plt.xlabel('Temperatura F')
#plt.ylabel('Affitti')
#plt.title('Visualizzazione dati originali')
#plt.show();

# Standardize the data
X_std = StandardScaler().fit_transform(df)

# Run local implementation of kmeans
km = KMeans(n_clusters=2, max_iter=1000)
km.fit(X_std)
centroids = km.cluster_centers_

# Plot the clustered data
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(X_std[km.labels_ == 0, 0], X_std[km.labels_ == 0, 1],
            c='green', label='cluster 1')
plt.scatter(X_std[km.labels_ == 1, 0], X_std[km.labels_ == 1, 1],
            c='blue', label='cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=300,
            c='r', label='centroid')
plt.legend()
#plt.xlim([-2, 2])
#plt.ylim([-2, 2])
plt.xlabel('Temperatura F')
plt.ylabel('Affitti')
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal')
plt.show();
