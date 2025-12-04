from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest

import nltk

import os
from quest.conf.settings import ABS_PROJECT_ROOT_PATH
from quest.core.chunker.splitter import spacyCutStep


nltk_user_path = os.path.join(ABS_PROJECT_ROOT_PATH, "model/nltk_data")
nltk.data.path.clear()
nltk.data.path.append(nltk_user_path)  # 将数据路径设置为当前目录下的data文件夹   

spacy_splitter = spacyCutStep()

def frequency_based_summary(text, num_sentences=5):
    # 分句
    sentences = sent_tokenize(text)
    
    # 分词并过滤停用词
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # 计算词频
    freq = FreqDist(words)
    
    # 计算句子重要性
    ranking = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if i in ranking:
                    ranking[i] += freq[word]
                else:
                    ranking[i] = freq[word]
    
    # 获取最重要的句子
    top_sentences = nlargest(num_sentences, ranking, key=ranking.get)
    summary = [sentences[j] for j in sorted(top_sentences)]
    
    return '\n'.join(summary)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def textrank_summary(text, num_sentences=5):
    sentences = sent_tokenize(text)
    
    # 预处理：分词、去停用词、词形还原等
    stop_words = set(stopwords.words('english'))
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    words = [[word for word in sentence if word.isalnum() and word not in stop_words] 
             for sentence in words]
    
    # 创建句子向量（基于词频）
    vocab = set([word for sentence in words for word in sentence])
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    sentence_vectors = []
    for sentence in words:
        vector = np.zeros(len(vocab))
        for word in sentence:
            vector[word_to_idx[word]] += 1
        sentence_vectors.append(vector)
    
    # 计算相似度矩阵
    sim_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_matrix[i][j] = cosine_similarity(
                    sentence_vectors[i].reshape(1, -1), 
                    sentence_vectors[j].reshape(1, -1)
                )[0, 0]
    
    # 构建图并计算PageRank
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    
    # 获取最重要的句子
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    summary = [sentence for score, sentence in ranked_sentences[:num_sentences]]
    
    return '\n'.join(summary)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

def deprecated_tfidf_summary(text, num_sentences=1):
    sentences =  sent_tokenize(text) # nltk
    # spacy_splitter.split_text(text)
    # print(f"文章句子总长度:{len(sentences)}")
    # a= input()
    
    # 计算TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    
    # 计算句子重要性（TF-IDF分数之和）
    sentence_scores = X.sum(axis=1)
    
    # 获取最重要的句子
    # 修正：将NumPy数组转换为整数列表
    top_indices = np.argsort(sentence_scores, axis=0).flatten()[-num_sentences:]
    top_indices = top_indices.tolist()[0]
    # print(top_indices)
    top_sentences = [sentences[i] for i in top_indices]
    
    return '\n'.join(top_sentences)

def tfidf_summary(text, num_sentences=1):
    sentences = sent_tokenize(text)
    
    # 验证输入
    if not sentences:
        return ""
    
    # 过滤空句子和过短的句子
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not valid_sentences:
        return sentences[0] if sentences else ""
    
    # 如果请求的句子数超过可用句子数，调整数量
    num_sentences = min(num_sentences, len(valid_sentences))
    
    try:
        # 计算TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,  # 确保至少出现1次的词被包含
            max_features=1000,  # 限制特征数量
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # 只匹配字母且长度>=2的词
        )
        X = vectorizer.fit_transform(valid_sentences)
        
        # 检查是否生成了有效的特征
        if X.shape[1] == 0:
            # 如果没有有效特征，返回第一个句子
            return valid_sentences[0]
        
        # 计算句子重要性（TF-IDF分数之和）
        sentence_scores = X.sum(axis=1).A1  # .A1 将矩阵转换为1D数组
        
        # 获取最重要的句子
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        
        # 按原文顺序排序索引
        top_indices = sorted(top_indices)
        
        top_sentences = [valid_sentences[i] for i in top_indices]
        
        return '\n'.join(top_sentences)
        
    except ValueError as e:
        if "empty vocabulary" in str(e):
            # 如果词汇表为空，返回第一个有效句子
            return valid_sentences[0]
        else:
            raise e


if __name__ == "__main__":

    text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and 
    human language, in particular how to program computers to process and analyze 
    large amounts of natural language data. The result is a computer capable of 
    "understanding" the contents of documents, including the contextual nuances of 
    the language within them. The technology can then accurately extract information 
    and insights contained in the documents as well as categorize and organize the 
    documents themselves. Challenges in natural language processing frequently involve 
    speech recognition, natural language understanding, and natural language generation.
    """

    print("基于词频的摘要:")
    print(frequency_based_summary(text, num_sentences=2))
    # print("\n基于TextRank的摘要:")
    # print(textrank_summary(text, num_sentences=2))
    print("\n基于TF-IDF的摘要:")
    print(tfidf_summary(text, num_sentences=2))

