
import numpy as np
import faiss


def faiss_kmeans_clustering(embeddings, n_clusters, seed=123):
    """
    用FAISS的KMeans对embeddings聚类
    :param embeddings: numpy array, shape (n_samples, embedding_dim)
    :param n_clusters: int, 聚类数量
    :param seed: int, 随机种子
    :return: (centroids, labels)
        - centroids: numpy array, shape (n_clusters, embedding_dim)
        - labels: numpy array, shape (n_samples,)
    """
    # FAISS 需要 float32 格式
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    kmeans = faiss.Kmeans(d=embeddings.shape[1], k=n_clusters, seed=seed, verbose=False)
    kmeans.train(embeddings)
    
    # 获取聚类中心
    centroids = kmeans.centroids
    
    # 获取每个embedding的聚类编号
    D, labels = kmeans.index.search(embeddings, 1)
    labels = labels.reshape(-1)
    
    return centroids, labels



def cluster_docs_by_embedding(doc_2_embedding, n_clusters=5, seed=123):
    """
    对 doc_2_embedding 进行聚类，返回 doc_id 到 cluster_id 的映射
    embedding是(1, d)的numpy.ndarray
    :param doc_2_embedding: dict, {doc_id(int): embedding}
    :param n_clusters: int, 聚类数量
    :param seed: int, 随机种子
    :return: dict, {doc_id: cluster_id}
    """
    doc_ids = list(doc_2_embedding.keys())
    embeddings = np.stack([doc_2_embedding[doc_id] for doc_id in doc_ids])
    _, labels = faiss_kmeans_clustering(embeddings, n_clusters, seed)
    doc_2_cluster_id = {doc_id: int(labels[i]) for i, doc_id in enumerate(doc_ids)}
    return doc_2_cluster_id

