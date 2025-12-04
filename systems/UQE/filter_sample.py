from f import func

from repository import ModelRepository
from gen_embeds import get_text_embeds, get_image_embeds
from online import online_sampling, online_without_sampling, online_sampling_merged, online_without_sampling_merged
from utils import read_and_concatenate, read_image, read_image_PIL

import os
import pandas as pd
import numpy as np

from config_uqe import BATCH_SIZE, BUDGET

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_repository = ModelRepository()


def gen_text_embedding(df, col_name):
    text_model = model_repository.get_text_model()
    df_col = df[col_name]
    col_embeds = get_text_embeds(df_col, text_model)
    return col_embeds

def gen_csv_embedding(df, col_name, data_schema):
    text_model = model_repository.get_text_model()
    df_col = df[col_name]
    new_df = []
    for filepath in df_col:
        new_df.append(read_and_concatenate(data_schema, filepath))
    new_df = pd.Series(new_df)
    col_embeds = get_text_embeds(new_df, text_model)
    return col_embeds

def gen_img_embedding(df, col_name, data_schema):
    image_model = model_repository.get_image_model()
    df_col = df[col_name]
    new_df = []
    for filepath in df_col:
        new_df.append(read_image_PIL(data_schema, filepath))
    col_embeds = get_image_embeds(new_df, image_model)
    return col_embeds

def sample_text_rows(df, col_name, col_attr, cond_val, cond_op, data_schema):
    print("Sampling text rows...")

    text_model = model_repository.get_text_model()
    df_col = df[col_name]
    col_embeds = get_text_embeds(df_col, text_model)
    n_samples = col_embeds.shape[0]
    n_features = col_embeds.shape[1]

    prompt = data_schema.get_filter_prompt(col_name, col_attr, cond_val, cond_op)

    # sampling
    print("Sampling rows based on the following prompt:")
    selected_row_index = online_without_sampling(df_col, col_embeds, prompt)
    return selected_row_index

def sample_csv_rows(df, col_name, col_attr, cond_val, cond_op, data_schema):

    text_model = model_repository.get_text_model()
    df_col = df[col_name]
    new_df = []
    for filepath in df_col:
        new_df.append(read_and_concatenate(data_schema, filepath))
    new_df = pd.Series(new_df)
    df[col_name] = new_df

    col_embeds = get_text_embeds(new_df, text_model)
    n_samples = col_embeds.shape[0]
    n_features = col_embeds.shape[1]

    # llm prompt条件
    prompt = data_schema.get_filter_prompt(col_name, col_attr, cond_val, cond_op)

    selected_row_index = online_without_sampling(df_col, col_embeds, prompt)
    return selected_row_index


def sample_image_rows(df, col_name, col_attr, cond_val, cond_op, data_schema):
    image_model = model_repository.get_image_model()
    df_col = df[col_name]

    new_df = []
    for filepath in df_col:
        new_df.append(read_and_concatenate(data_schema, filepath))
    new_df = pd.Series(new_df)
    col_embeds = get_image_embeds(new_df, image_model)
    n_samples = col_embeds.shape[0]
    n_features = col_embeds.shape[1]

    prompt = data_schema.get_filter_prompt(col_name, col_attr, cond_val, cond_op)

    selected_row_index = online_without_sampling(new_df, col_embeds, prompt)
    return selected_row_index

def sample_merged_rows(df, col_to_add, data_schema):
    df_copy = df.copy(deep=True)
    col_names = list(col_to_add.keys())
    full_embeds = None
    n_total_features = 0
    for col_name in col_names:
        col_type = data_schema.get_col_type(col_name)
        if col_type == 'text':
            df_col = df[col_name]
            text_model = model_repository.get_text_model()
            col_embeds = get_text_embeds(df_col, text_model)
        elif col_type == 'csv':
            df_col = df[col_name]
            new_df = []
            for filepath in df_col:
                new_df.append(read_and_concatenate(data_schema, filepath))
            new_df = pd.Series(new_df)
            text_model = model_repository.get_text_model()
            col_embeds = get_text_embeds(new_df, text_model)
            df_copy[col_name] = new_df
        elif col_type == 'image':
            df_col = df[col_name]
            new_df = []
            for filepath in df_col:
                new_df.append(read_image_PIL(data_schema, filepath))
            image_model = model_repository.get_image_model()
            col_embeds = get_image_embeds(new_df, image_model)

        n_samples = col_embeds.shape[0]
        n_features = col_embeds.shape[1]
        n_total_features += n_features
        if full_embeds is None:
            full_embeds = col_embeds
        else:
            assert full_embeds.shape[0] == n_samples
            full_embeds = np.hstack((full_embeds, col_embeds))
    
    prompt = data_schema.get_merged_filter_prompt(col_to_add)
    selected_row_index = online_sampling_merged(df_copy, col_names, full_embeds, prompt, int(n_samples * BUDGET), batch_size=BATCH_SIZE)

    return selected_row_index
