from repository import ModelRepository

import torch
import numpy as np
import pandas as pd
from sentence_transformers import util
from PIL import Image

import config_uqe

repository = ModelRepository()
batch_size = 4

def get_text_embeds(sentences, text_model):
    text_model = repository.get_text_model()
    with torch.no_grad():
        if config_uqe.USE_BART:
            tokenizer, model = text_model
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
            sentence_embeddings = model(inputs['input_ids'].to(model.device),
                                        inputs["attention_mask"].to(model.device)).last_hidden_state.mean(dim=1)
        else:
            sentence_embeddings = text_model.encode(sentences, convert_to_tensor=True, device=text_model.device)
    sentence_embeddings = util.normalize_embeddings(sentence_embeddings)
    return sentence_embeddings.detach().cpu().numpy()

def get_image_embeds(images, image_model):
    model, preprocess, t = image_model
    with torch.no_grad():
        inputs = preprocess(images=images, return_tensors="pt")
        image_embeddings = model.get_image_features(inputs.pixel_values.to(model.device))
    image_embeddings = util.normalize_embeddings(image_embeddings)
    return image_embeddings.detach().cpu().numpy()
