

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50,  ResNet50_Weights
from PIL import Image
import os
import json
import urllib
import json
import urllib.request
import numpy as np

# Definisci le trasformazioni per pre-elaborare le immagini
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Funzione per caricare e pre-elaborare l'immagine
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Aggiungi una dimensione per il batch
    return img_tensor

def process_image(img):
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Aggiungi una dimensione per il batch
    return img_tensor


# Funzione per calcolare l'entropia marginale
def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

def kl_div(outputs):
    kl_divs = []
    for i in range(outputs.size(1)):
        class_probs = outputs[:, i]
        class_avg_prob = class_probs.mean(dim=0)
        kl_div = (class_probs * (torch.log(class_probs) - torch.log(class_avg_prob))).sum(dim=-1)
        kl_divs.append(kl_div)
    
    # Somma delle KL Divergence
    total_kl_div = torch.stack(kl_divs).sum()
    return total_kl_div

# Carica il file di mapping delle classi di ImageNet
mapping_file_path = 'classes.txt'
class_mapping = {}
with open(mapping_file_path, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split(' ')
        class_mapping[parts[0]] = ' '.join(parts[1:])

# Funzione per ottenere l'etichetta leggibile dal ID della classe
def get_class_label(class_id):
    return class_mapping.get(class_id, "Unknown class")
