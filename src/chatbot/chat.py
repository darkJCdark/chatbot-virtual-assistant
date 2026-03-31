import torch
from torch.nn import functional as F
import tiktoken # <-- AGREGADO: Importamos tiktoken

# Asegúrate de tener tu modelo cargado, la variable 'device' definida, 
# y el modelo en modo evaluación antes de ejecutar el bucle
# model.eval() # <-- ¡MUY IMPORTANTE PARA QUE NO FALLE!

@torch.no_grad()
def generate_text(model, idx, max_new_tokens, temperature=0.8, top_k=40, block_size=256):
    """
    Genera texto nuevo usando Temperature y Top-K para evitar repeticiones.
    """
    for _ in range(max_new_tokens):
        # Recortamos el contexto si excede el tamaño máximo del bloque
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Obtenemos las predicciones (logits)
        logits, _ = model(idx_cond)
        
        # Nos enfocamos solo en el último paso de tiempo (la siguiente palabra)
        logits = logits[:, -1, :] 
        
        # 1. TEMPERATURE: Suavizamos o afilamos las probabilidades
        logits = logits / temperature
        
        # 2. TOP-K: Nos quedamos solo con las 'k' mejores opciones, ignorando el resto
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Convertimos a probabilidades usando softmax
        probs = F.softmax(logits, dim=-1)
        
        # 3. MUESTREO: Elegimos la siguiente palabra basándonos en las probabilidades
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Añadimos la nueva palabra a la secuencia
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# ==========================================
# BUCLE DE CHAT CORREGIDO
# ==========================================

# <-- AGREGADO: Inicializamos el tokenizador (el mismo que usamos en el entrenamiento)
enc = tiktoken.get_encoding("gpt2")

print("\n✅ Modelo cargado con éxito. (Escribe 'salir' para terminar)")

while True:
    try:
        user_input = input("\n👤 Tú: ")
        if user_input.lower() in ['salir', 'exit', 'quit']:
            break
            
        # Pásale el texto crudo tal cual, sin "User:" ni "Assistant:"
        input_ids = enc.encode(user_input)
        
        # Asumo que la variable 'device' ya está definida más arriba en tu script original
        # (ej. device = 'cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        print("\n--- La IA está pensando... ---")
        
        # Generamos unos 100 tokens nuevos
        output_tensor = generate_text(model, input_tensor, max_new_tokens=100, temperature=0.8, top_k=40)
        
        # Decodificamos de vuelta a texto
        respuesta_texto = enc.decode(output_tensor[0].tolist())
        
        print(f"\n🤖 IA: {respuesta_texto}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Generación detenida por el usuario.")