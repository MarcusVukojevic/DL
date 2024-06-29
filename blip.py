from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Carica il processore e il modello
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Carica un'immagine di esempio
url = "https://huggingface.co/front/thumbnails/transformers.png"
image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("imagenet-a/n07749582/0.000017_broccoli _ broccoli_0.52701503.jpg")
# Prepara l'immagine e genera la descrizione
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)

# Stampa la descrizione generata
print(processor.decode(out[0], skip_special_tokens=True))
