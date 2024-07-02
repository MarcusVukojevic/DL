from augmentations import ImageAugmentor
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.transforms import transforms
from utils import process_image, marginal_entropy, load_and_preprocess_image, get_class_label, kl_div
import os 
import torch
import urllib, json
from copy import deepcopy
from tqdm import tqdm

from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests

# Carica il processore e il modello per VQA
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_blip = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

augmentator = ImageAugmentor()
n_augmentations = 20

device = torch.device("mps:0" if torch.has_mps else "cpu")

# Scarica il file di etichette delle classi di ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = urllib.request.urlopen(LABELS_URL)
class_idx = json.loads(response.read().decode())

# Carica il modello ResNeXt101
model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

pesi_modello = deepcopy(model.state_dict())

train_mean = []
train_var = []
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        train_mean.append(deepcopy(module.running_mean))
        train_var.append(deepcopy(module.running_var))

# Percorso alla cartella delle immagini

numero_immagini_totali = 0
numero_immagini_cls_corrette = 0

for cartella in tqdm(os.listdir("imagenet-a"), desc="Macinando classi"):

    image_folder = f'imagenet-a/{cartella}'
    
    if os.path.isdir(image_folder) == False:
        continue
    true_label = get_class_label(image_folder[-9:])
    # Lista delle immagini nella cartella
    image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')]

    numero_immagini_totali += len(image_files)

    dio = 0
    numero_uguali = 0

    N = 16 # --> questo per la bn

    for i in image_files:
        
        model.train()

        immagini_aug = augmentator.apply_all_augmentations(i)
        lista_probabilità = []

        optimizer.zero_grad()
        
        predizioni_tmp = []

        image_test = Image.open(f"{i}")

        dizionario_blip = {}

        for img in immagini_aug:
            proc = process_image(img).to(device) # immagine processata
            output = model(proc)
            
            _, preds = torch.max(output, 1)
            predizioni_tmp.append(class_idx[preds.item()])

            if class_idx[preds.item()] in dizionario_blip:
                if dizionario_blip[class_idx[preds.item()]] in ["yes", "there is", "correct", "true"]:
                    output[0][preds.item()] += output[0][preds.item()] * 2
            else:
                question = f"Is there a {class_idx[preds.item()]} in the picture?"

                inputs = processor_blip(image_test, question, return_tensors="pt")
                out = model_blip.generate(**inputs, max_length = 2)
                risposta_blip = processor_blip.decode(out[0], skip_special_tokens=True)

                dizionario_blip[class_idx[preds.item()]] = risposta_blip.lower()

                if risposta_blip.lower() in ["yes", "there is", "correct", "true"]:
                    output[0][preds.item()] += output[0][preds.item()] * 2
                
            lista_probabilità.append(output)
       
        indice = 0
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                mu_train = train_mean[indice]
                var_train = train_var[indice]
                mu_test = module.running_mean
                var_test = module.running_var
                
                module.running_mean = (N / (N + 1)) * mu_train + (1 / (N + 1)) * mu_test
                module.running_var = (N / (N + 1)) * var_train + (1 / (N + 1)) * var_test

                indice += 1

        probabilities_tensor = torch.stack(lista_probabilità)
        mean_entropy, avg_logits = marginal_entropy(probabilities_tensor)

        mean_entropy.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            output = model(load_and_preprocess_image(i).to(device))
            _, preds = torch.max(output, 1)

            if class_idx[preds.item()] == true_label:
                numero_uguali += 1
                numero_immagini_cls_corrette += 1

        model.load_state_dict(pesi_modello)

        dio += 1
    
    print(f"il numero d immagini che ho classificato come uguali per la classe ----> {true_label}: {numero_uguali}")

print(f"Classificate correttamente {numero_immagini_cls_corrette/numero_immagini_totali}")
print(f"Numero immagini totali: {numero_immagini_totali}")
print(f"Numero immagini classificate correttamente: {numero_immagini_cls_corrette}")
