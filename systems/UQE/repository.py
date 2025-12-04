from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, BartTokenizer, BartModel
import torch
from torchvision import transforms

import config_uqe


class ModelRepository:
    def __init__(self):
        print("Loading models...")
        self.models = {}
        self.os = "macos"
        if self.os == "macos":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device_id = torch.cuda.current_device() if torch.cuda.is_available() else -1
        # text model
        # if use local model, set the path here
        self.text_model_path = 'all-MiniLM-L6-v2'
        self.text_bart_model_path = 'bart-large-mnli'
        self.image_clip_model_path = 'clip-vit-base-patch32'

    def get_text_model(self):
        if 'text' not in self.models:
            if config_uqe.USE_BART:
                tokenizer = BartTokenizer.from_pretrained(self.text_bart_model_path)
                model = BartModel.from_pretrained(self.text_bart_model_path)
                if self.os == "macos":
                    model = model.to(self.device)
                else:
                    if self.device_id >= 0:
                        model = model.to(self.device_id)
                self.models['text'] = (tokenizer, model)
            else:
                self.models['text'] = SentenceTransformer(self.text_model_path)
                if self.os == "macos":
                    self.models['text'] = self.models['text'].to(self.device)
                else:
                    if self.device_id >= 0:
                        self.models['text'] = self.models['text'].to(self.device_id)
        return self.models['text']

    def get_image_model(self):
        if 'image' not in self.models:
            model = CLIPModel.from_pretrained(self.image_clip_model_path)
            if self.os == "macos":
                model = model.to(self.device)
            else:
                if self.device_id >= 0:
                    model = model.to(self.device_id)
            preprocess = CLIPProcessor.from_pretrained(self.image_clip_model_path)
            t = transforms.Compose([transforms.ToPILImage()])
            self.models['image'] = (model, preprocess, t) 
        return self.models['image']
    
    def get_gpt_model(self):
        if 'image' not in self.models:
            self.models['exec'] = OpenAI()
        return self.models['exec']
