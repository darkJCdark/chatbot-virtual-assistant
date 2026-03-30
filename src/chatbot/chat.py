import os
import torch
import tiktoken
from torch.nn import functional as F
# Importamos la arquitectura desde tu archivo de entrenamiento
# (Asegúrate de que la clase MiniGPT y Block sean iguales)
from src.models.training import MiniGPT, n_embd, n_layer, n_head, block_size, device

def generate_response(model, enc, prompt, max_new_tokens=150, temperature=0.8):
    model.eval()
    # Convertimos tu pregunta a números
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    print("\n--- La IA está pensando... ---\n")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Recortamos el contexto si es muy largo
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            # Aplicamos "Temperature" para que no sea tan repetitiva
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Si genera el token de fin de texto (opcional) o queremos parar:
            if idx_next.item() == 50256: # Token <|endoftext|> de GPT2
                break

    return enc.decode(idx[0].tolist())

# --- INICIO DEL PROGRAMA ---
checkpoint_path = "src/models/checkpoints/model_best.pth"

if not os.path.exists(checkpoint_path):
    print(f"❌ Error: No encontré el archivo {checkpoint_path}")
else:
    # 1. Cargar cerebro
    enc = tiktoken.get_encoding("gpt2")
    model = MiniGPT().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Modelo cargado con éxito. ¡Ya puedes hablar con tu IA!")

    # 2. Bucle de Chat
    while True:
        user_input = input("\n👤 Tú: ")
        if user_input.lower() in ['salir', 'exit', 'quit']: break
        prompt_estructurado = f"User: {user_input}\nAssistant:"
        response = generate_response(model, enc, prompt_estructurado)
        print(f"🤖 IA: {response}")