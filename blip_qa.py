from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests

# Carica il processore e il modello per VQA
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Carica un'immagine di esempi

# Domanda sull'immagine
question = "Is there a hamburger in the picture?"
image = Image.open("imagenet-a/n07697313/0.003802_digital clock _ digital clock_0.75967115.jpg")

# Prepara i dati
inputs = processor(image, question, return_tensors="pt")
# Genera la risposta
out = model.generate(**inputs)

# Stampa la risposta
print(processor.decode(out[0], skip_special_tokens=True))
