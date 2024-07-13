
from augmentations import ImageAugmentor
from torchvision.models import resnet50, ResNet50_Weights
from utils import process_image, marginal_entropy, load_and_preprocess_image, get_class_label, kl_div
import os 
import torch
import urllib, json
from copy import deepcopy
from tqdm import tqdm

augmentator = ImageAugmentor()
n_augmentations = 64


device = torch.device("mps:0" if torch.has_mps else "cpu")

# Scarica il file di etichette delle classi di ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = urllib.request.urlopen(LABELS_URL)
class_idx = json.loads(response.read().decode())

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
# weights=models.ResNet50_Weights.IMAGENET1K_V1
# weights=weights=ResNet50_Weights.DEFAULT
model.train() 

#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
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

        immagini_aug = augmentator.apply_augmentations(i, True, n_augmentations)
        lista_probabilità = []

        optimizer.zero_grad()
        
        #print("PREDIZIONE AUG:")
        for img in immagini_aug:
            proc = process_image(img).to(device) # immagine processata
            output = model(proc)
            _, preds = torch.max(output, 1)
            #print(class_idx[preds.item()])
            #probabilities = torch.nn.functional.softmax(output[0], dim=0)
            lista_probabilità.append(output)
       
        indice = 0
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # Recupera le statistiche attuali
                mu_train = train_mean[indice]
                var_train = train_var[indice]
                # Calcola le statistiche del test
                mu_test = module.running_mean
                var_test = module.running_var
                
                # Mescola le statistiche
                module.running_mean = (N / (N + 1)) * mu_train + (1 / (N + 1)) * mu_test
                module.running_var = (N / (N + 1)) * var_train + (1 / (N + 1)) * var_test

                indice = indice + 1

        # Converti la lista in un tensor e calcola la media
        probabilities_tensor = torch.stack(lista_probabilità)
        mean_entropy, avg_logits = marginal_entropy(probabilities_tensor) # da mean_entropy.item()

        # Backpropagazione e aggiornamento dei parametri
        mean_entropy.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            output = model(load_and_preprocess_image(i).to(device))
            _, preds = torch.max(output, 1)

            if (class_idx[preds.item()] == true_label):
                numero_uguali += 1
                numero_immagini_cls_corrette += 1

        model.load_state_dict(pesi_modello)

        dio += 1
    print(f"il numero d immagini che ho classificato come uguali per la classe ----> {true_label}: {numero_uguali}")

print(f"Classificate correttamente {numero_immagini_cls_corrette/numero_immagini_totali}")
print(f"Numero immagini totali: {numero_immagini_totali}")
print(f"Numero immagini classificate correttamente: {numero_immagini_cls_corrette}")