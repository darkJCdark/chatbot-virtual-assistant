import os
import torch
import tiktoken
from src.models.model import MiniGPT
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")
nombre_ia = "Hannah"

model = MiniGPT(enc.n_vocab, device).to(device)
checkpoint = torch.load('src/models/checkpoints/hannah_finetuned.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Historial gestionado por turnos enteros, no por límite de caracteres
historial_lista = []

print(f"💖 {nombre_ia} lista. (Escribe 'salir' para terminar)")

while True:
    msg = input("\n👤 Tú: ")
    if msg.lower() in ['salir', 'exit']: break
    
    # 1. Alineación estricta con el formato exacto de entrenamiento
    header = f"Context: I am {nombre_ia}, your girlfriend."
    
    historial_str = ""
    # Mantenemos solo los últimos 4 turnos para no sobrecargar el contexto de 256
    for h_msg in historial_lista[-4:]: 
        historial_str += f"{h_msg} "
        
    prompt = f"{header} | {historial_str}User: {msg} {nombre_ia}:"
    idx = torch.tensor([enc.encode(prompt, allowed_special={'<|endoftext|>'})], dtype=torch.long).to(device)
    
    # 2. Generación con Top-K y Top-P (Nucleus Sampling) integrados
    for _ in range(80):
        logits, _ = model(idx[:, -256:])
        next_token_logits = logits[:, -1, :] / 0.8 # Temperatura
        
        # Filtro Top-K (Mantiene solo las 50 opciones más probables)
        top_k = 50
        v, _ = torch.topk(next_token_logits, top_k)
        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
        
        # Filtro Top-P (Nucleus Sampling al 90%)
        top_p = 0.9
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = -float('Inf')
        
        # Muestreo final
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, next_token), dim=1)
        if next_token.item() == 50256: break # <|endoftext|>
    
    # Decodificación limpia
    respuesta_cruda = enc.decode(idx[0].tolist())
    respuesta = respuesta_cruda.split(f"{nombre_ia}:")[-1].replace("<|endoftext|>", "").strip()
    print(f"\n🌸 {nombre_ia}: {respuesta}")
    
    # Almacenamiento seguro en la lista de historial
    historial_lista.append(f"User: {msg}")
    historial_lista.append(f"{nombre_ia}: {respuesta}")