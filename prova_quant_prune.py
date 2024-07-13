import torch
import torch.nn.utils.prune as prune
from transformers import BlipForQuestionAnswering

# Funzione per applicare la quantizzazione dinamica ai livelli Linear
def apply_dynamic_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Quantizza dinamicamente il modulo Linear
            quantized_module = torch.quantization.quantize_dynamic(
                module, {torch.nn.Linear}, dtype=torch.qint8
            )
            # Ottieni il nome del modulo padre
            parent_name = '.'.join(name.split('.')[:-1])
            # Imposta il modulo quantizzato nel modello
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, name.split('.')[-1], quantized_module)
            else:
                model = quantized_module
    return model

# Funzione per applicare il pruning ai livelli Linear
def apply_pruning(model, amount=0.4):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

# Carica il modello pre-addestrato BLIP
model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')

# Imposta il modello in modalità valutazione
model.eval()

# Step 1: Applica la quantizzazione dinamica
torch.backends.quantized.engine = 'qnnpack'  # o 'fbgemm' se hai una CPU Intel
model = apply_dynamic_quantization(model)

# Step 2: Salva il modello quantizzato
torch.save(model.state_dict(), 'quantized_blip_model.pth')

# Step 3: Ricarica il modello quantizzato
model.load_state_dict(torch.load('quantized_blip_model.pth'))

# Step 4: Applica il pruning
model = apply_pruning(model, amount=0.4)

# Step 5: Salva il modello quantizzato e prunato
torch.save(model.state_dict(), 'quantized_pruned_blip_model.pth')

# Step 6: Ricarica il modello quantizzato e prunato per l'inferenza
model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')
model.load_state_dict(torch.load('quantized_pruned_blip_model.pth'), strict=False)

# Imposta il modello in modalità valutazione
model.eval()

