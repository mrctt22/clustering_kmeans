import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carica i dati dal file CSV
def load_data(csv_file):
    return pd.read_csv(csv_file, sep=";")

# Funzione per visualizzare i risultati del clustering
def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red')
    plt.title('Clustering dei dati')
    plt.xlabel('Temperatura')
    plt.ylabel('Affitti')
    plt.show()

# Esegui il clustering
def perform_clustering(data, num_clusters, num_iters):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=num_iters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids

def main():
    # Carica i dati dal file CSV
    csv_file = r".\data.csv"
    data = load_data(csv_file)

    # Effettua il clustering dei dati
    num_clusters = 3  # Numero di cluster desiderato
    num_iters = 10 # Numero di iterazioni
    labels, centroids = perform_clustering(data, num_clusters, num_iters)

    # Visualizza i risultati del clustering
    plot_clusters(data.values, labels, centroids)

if __name__ == "__main__":
    main()