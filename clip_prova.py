import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests

# Carica il processore e il modello CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Carica un'immagine di esempio
url = "https://huggingface.co/front/thumbnails/transformers.png"
#image = Image.open(requests.get(url, stream=True).raw)

image = Image.open("imagenet-a/n07753592/0.004971_syringe _ stethoscope_0.14998975.jpg")
# Definisci l'oggetto da cercare
object_to_find = "There is a coconut in the image"

# Prepara i dati
inputs = processor(text=[object_to_find], images=image, return_tensors="pt", padding=True)

# Calcola le similitudini
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # il logit dell'immagine rispetto al testo
probs = logits_per_image.softmax(dim=1) # converti in probabilità
print(probs)
# Stampa la probabilità che l'oggetto sia presente
print(f"Probability that the object '{object_to_find}' is present in the image: {probs.item() * 100:.2f}%")
