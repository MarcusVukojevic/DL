import torch
from transformers import BlipForQuestionAnswering

# Step 1: Carica il modello pre-addestrato BLIP
model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')

# Step 2: Imposta il motore di quantizzazione preferito
torch.backends.quantized.engine = 'qnnpack'  # o 'fbgemm' se hai una CPU Intel

# Step 3: Applica la quantizzazione dinamica
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Step 4: Salva il modello quantizzato
torch.save(quantized_model.state_dict(), 'quantized_blip_model.pth')

# -- Successivamente, quando vuoi caricare il modello quantizzato --

# Step 5: Carica la struttura del modello
model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')

# Step 6: Carica i pesi quantizzati
quantized_model_state_dict = torch.load('quantized_blip_model.pth')

# Verifica la struttura del modello e i pesi caricati
model.load_state_dict(quantized_model_state_dict, strict=False)

# Step 7: Imposta il modello in modalità valutazione
model.eval()

# Il modello è ora pronto per essere utilizzato per l'inferenza
