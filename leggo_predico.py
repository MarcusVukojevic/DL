
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50,  ResNet50_Weights
from PIL import Image
import os
import json
import urllib
import json
import urllib.request
from transformers import CLIPProcessor, CLIPModel


# Carica il modello CLIP pre-addestrato e il processore
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Carica il modello ResNet50 pre-addestrato

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# Definisci le trasformazioni per pre-elaborare le immagini
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Percorso alla cartella delle immagini
image_folder = 'imagenet-a/n07753592'

# Lista delle immagini nella cartella
image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')]


# Carica il file di mapping delle classi di ImageNet
mapping_file_path = 'classes.txt'
class_mapping = {}
with open(mapping_file_path, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split(' ')
        class_mapping[parts[0]] = ' '.join(parts[1:])


# Scarica il file di etichette delle classi di ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = urllib.request.urlopen(LABELS_URL)
class_idx = json.loads(response.read().decode())



# Funzione per caricare e pre-elaborare l'immagine
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Aggiungi una dimensione per il batch
    return img_tensor

# Funzione per ottenere l'etichetta leggibile dal ID della classe
def get_class_label(class_id):
    return class_mapping.get(class_id, "Unknown class")

# Test con l'ID della classe
class_id = 'n07753592'
true_label = get_class_label(class_id)
print(f"The true label for class ID {class_id} is: {true_label}")


# Prevedi su ciascuna immagine
predictions = {}
for img_path in image_files:
    img_tensor = load_and_preprocess_image(img_path)
    img_clip = Image.open(img_path).convert("RGB")
    
    with torch.no_grad():
        outputs = model(img_tensor)


    _, preds = torch.max(outputs, 1)

    texts = [f"a photo of a {i}" for i in class_idx]
    inputs = processor(text=texts, images=img_clip, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model_clip.get_image_features(inputs['pixel_values'])
        image_features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        text_features = model_clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Calcola il punteggio di similarità coseno
    similarity = torch.matmul(image_features, text_features.T)
    probabilities = torch.nn.functional.softmax(similarity, dim=1)

    # Ottieni l'indice della classe con la probabilità maggiore
    _, max_index = torch.max(probabilities, dim=1)

    # Stampa la classe con la probabilità maggiore
    predicted_class = class_idx[max_index.item()]

    text = f"a photo of a {true_label}"
    inputs = processor(text=[text], images=img_clip, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model_clip.get_image_features(inputs['pixel_values'])
        image_features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        text_inputs = processor(text=[text], return_tensors="pt", padding=True)
        text_features = model_clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Calcola il punteggio di similarità coseno
    similarity = torch.matmul(image_features, text_features.T)
    score_vero = similarity.item()

    predictions[img_path] = [preds.item(), predicted_class, score_vero]


# Mappa gli indici delle classi alle etichette leggibili
for img_path, pred in predictions.items():
    label = class_idx[pred[0]]
    print(f"Predictions: {label}, true label: {true_label}, clip_falso: {pred[1]}, clip_vero: {pred[2]}")

