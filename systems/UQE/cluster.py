import faiss
import pickle
import numpy as np

from f import func, all_one, df_check_nan, df_check_nan_sum, llm_filter_aggr_merged_split
from filter_sample import gen_text_embedding, gen_csv_embedding, gen_img_embedding

from utils import read_and_concatenate, read_image

from config_uqe import AGGR_CLUSTER_SAMPLE_RATIO


class Aggregator:
    def __init__(self, ncentroids, d, niter=40, verbose=True, gpu=True, 
        ratio_samples=AGGR_CLUSTER_SAMPLE_RATIO, straitegy='proportional'):
        self.ncentroids = ncentroids
        self.d = d
        self.niter = niter
        self.verbose = verbose
        self.gpu = gpu
        self.kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=gpu)

        self.ratio_samples = ratio_samples
        self.straitegy = straitegy  
 
    def fit(self, x):
        self.kmeans.train(x)
        self.cluster_cents = self.kmeans.centroids
        self.cluster_wucha = self.kmeans.obj
        self.D, self.I = self.kmeans.index.search(x, 1)
        self.n = x.shape[0]
 
    def get_cluster_dict(self):
        cluster_dict = {}
        for i in range(self.ncentroids):
            cluster_dict[i] = np.where(self.I == i)[0]
        return cluster_dict
    
    def stratified_sample(self, cluster_dict):
        n_samples_list = []
        if self.straitegy == 'equal':
            for i in range(self.ncentroids):
                n_samples_list.append(int(self.n * self.ratio_samples / self.ncentroids))
        elif self.straitegy == 'proportional':
            for i in range(self.ncentroids):
                n_samples_list.append(int(len(cluster_dict[i]) * self.ratio_samples))

        sample_idx = []
        sampled_cluster_dict = {}
        for i in range(self.ncentroids):
            idx = cluster_dict[i]
            n_samples = n_samples_list[i]
            if len(idx) < n_samples:
                idx = np.random.choice(idx, n_samples, replace=True)
            else:
                idx = np.random.choice(idx, n_samples, replace=False)
            sampled_cluster_dict[i] = idx
            sample_idx.extend(idx)
        return sampled_cluster_dict
    
    def compute_weights(self, cluster_dict, sampled_cluster_dict):
        sample_idx = []
        for key in sampled_cluster_dict.keys():
            sample_idx.extend(sampled_cluster_dict[key])

        # compute cluster sizes
        cluster_sizes = []
        for i in range(self.ncentroids):
            cluster_sizes.append(len(cluster_dict[i]))

        sampled_cluster_sizes = []
        for i in range(self.ncentroids):
            sampled_cluster_sizes.append(len(sampled_cluster_dict[i]))

        # compute weights
        weights = np.zeros(len(sample_idx))
        for i, idx_i in enumerate(sample_idx):
            denom_i = 0
            cluster_idx_i = int(self.I[idx_i][0])
            denom_i = sampled_cluster_sizes[cluster_idx_i]
            weights[i] = cluster_sizes[cluster_idx_i] / denom_i
        return weights

    def count_aggregator(self, x, sample_idx, weights, func=all_one):
        expectation = 0
        sum_weights = sum(weights)
        for i, idx in enumerate(sample_idx):
            expectation += weights[i] / sum_weights * func(x[idx], None)

        print(expectation)
        return self.n * expectation
    
    def count_aggregator_df(self, df, weights, func=df_check_nan):
        expectation = 0
        sum_weights = sum(weights)
        for i in range(len(df)):
            expectation += weights[i] / sum_weights * func(df.iloc[i])

        print(expectation)
        return self.n * expectation
    
    def count_aggregator_df_test(self, df, weights, data_schema, 
        filter_columns_attrs=None, filter_sys_prompt=None, filter_sys_prompt_split=None):
        expectation = 0
        sum_weights = sum(weights)
        for i in range(len(df)):
            if filter_sys_prompt is None:
                expectation += weights[i] / sum_weights * func(df.iloc[i])
            else:
                user_prompt = []
                user_prompt_cell_list = []
                col_names = list(filter_columns_attrs.keys()) if filter_columns_attrs else []
                for col_name in col_names:
                    col_type = data_schema.get_col_type(col_name)
                    if col_type == "csv":
                        filepath = df.iloc[i]
                        cell = read_and_concatenate(data_schema, filepath)
                    elif col_type == "image":
                        filepath = df.iloc[i]
                        cell = read_image(data_schema, filepath)
                    else: # "text"
                        cell = df[col_name].iloc[i]
                    if col_type != "image":
                        user_prompt_cell = {
                            "role": "user",
                            "content": "The paragraph of '" + col_name + "': " + cell
                        }
                        user_prompt_cell_list.append(user_prompt_cell)
                        user_prompt.append(user_prompt_cell)
                    else:
                        user_prompt_cell = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{cell}"
                                    }
                                }
                            ]
                        }
                        user_prompt_cell_list.append(user_prompt_cell)
                        user_prompt.append(user_prompt_cell)
                
                user_prompt_dict = dict(zip(col_names, user_prompt))    
                filter_result = llm_filter_aggr_merged_split(user_prompt_dict, user_prompt_cell_list, filter_sys_prompt_split)

                if filter_result == 1:
                    expectation += weights[i] / sum_weights * func(df.iloc[i])
                else:
                    expectation += 0
        print(expectation)
        return self.n * expectation
    
    def sum_aggregator(self, x, x_sum, sample_idx, weights, func):
        expectation = 0
        sum_weights = sum(weights)
        for i, idx in enumerate(sample_idx):
            expectation += x_sum[i] * weights[i] / sum_weights * func(x[idx], None)

        print(expectation)
        return self.n * expectation
    
    def sum_aggregator_df(self, df, weights, func=df_check_nan_sum):
        expectation = 0
        sum_weights = sum(weights)
        for i in range(len(df)):
            result = func(df.iloc[i])
            if isinstance(result, float) or isinstance(result, int):
                expectation += weights[i] / sum_weights * result
            else:
                expectation += 0

        print(expectation)
        return self.n * expectation

    def sum_aggregator_df_test(self, df, col, weights, data_schema, func=df_check_nan_sum, 
        filter_columns_attrs=None, filter_sys_prompt=None, filter_sys_prompt_split=None):
        expectation = 0
        sum_weights = sum(weights)
        for i in range(len(df)):
            if filter_sys_prompt is None:
                expectation += weights[i] / sum_weights * func(df[col].iloc[i])
            else:
                user_prompt = []
                user_prompt_cell_list = []
                col_names = list(filter_columns_attrs.keys()) if filter_columns_attrs else []
                for col_name in col_names:
                    col_type = data_schema.get_col_type(col_name)
                    if col_type == "csv":
                        filepath = df.iloc[i]
                        cell = read_and_concatenate(data_schema, filepath)
                    elif col_type == "image":
                        filepath = df.iloc[i]
                        cell = read_image(data_schema, filepath)
                    else: # "text"
                        cell = df[col_name].iloc[i]

                    if col_type != "image":
                        user_prompt_cell = {
                            "role": "user",
                            "content": "The paragraph of '" + col_name + "': " + cell
                        }
                        user_prompt_cell_list.append(user_prompt_cell)
                        user_prompt.append(user_prompt_cell)
                    else:
                        user_prompt_cell = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "The image of '" + col_name + "' will be listed below. "
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{cell}"
                                    }
                                }
                            ]
                        }
                        user_prompt_cell_list.append(user_prompt_cell)
                        user_prompt.append(user_prompt_cell)
                
                user_prompt_dict = dict(zip(col_names, user_prompt))
                filter_result = llm_filter_aggr_merged_split(user_prompt_dict, user_prompt_cell_list, filter_sys_prompt_split)

                if filter_result == 1:
                    expectation += weights[i] / sum_weights * func(df[col].iloc[i])
                else:
                    expectation += 0

        print(expectation)
        return self.n * expectation

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

def gen_embeds(df, col_name, col_type, data_schema):
    if col_type == 'text':
        col_embeds = gen_text_embedding(df, col_name)
    elif col_type == 'csv':
        col_embeds = gen_csv_embedding(df, col_name, data_schema)
    elif col_type == 'image':
        col_embeds = gen_img_embedding(df, col_name, data_schema)
    return col_embeds
