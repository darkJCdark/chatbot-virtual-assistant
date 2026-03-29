import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import tiktoken

# ==========================================
# 1. HIPERPARÁMETROS (Ajustados para tu RTX 5070 Ti)
# ==========================================
batch_size = 32       # Cuántos ejemplos procesa en paralelo
block_size = 64       # Contexto máximo (cuántas palabras recuerda hacia atrás)
max_iters = 1000      # Pasos de entrenamiento (va a tardar unos segundos/minutos)
eval_interval = 100   # Cada cuánto imprimimos el progreso
learning_rate = 1e-3
device = 'cuda' 

# Parámetros del Transformer
n_embd = 64           # Tamaño del vector de cada palabra
n_layer = 4           # Cuántas "capas de razonamiento" tiene
n_head = 4            # Cuántas cabezas de atención (multitarea)

print(f"--- Arrancando Motor en: {device.upper()} ---")

# ==========================================
# 2. DATA PIPELINE (El que ya probaste con éxito)
# ==========================================
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

def encode_todo(example):
    return {'ids': enc.encode(example['text'])}

tokenized_data = dataset.map(encode_todo, remove_columns=['text'], desc="Tokenizando")
todos_los_tokens = [id for fila in tokenized_data for id in fila['ids']]
data_tensor = torch.tensor(todos_los_tokens, dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i : i + block_size] for i in ix])
    y = torch.stack([data_tensor[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 3. LA ARQUITECTURA: MINI-TRANSFORMER (Estilo GPT)
# ==========================================
class Block(nn.Module):
    """ Un bloque de Transformer: Comunicación (Attention) + Computación (FeedForward) """
    def __init__(self):
        super().__init__()
        # La "Atención" (PyTorch ya tiene esto optimizado)
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, batch_first=True)
        # La "Reflexión"
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Creamos una máscara para que no haga trampa mirando al futuro
        mask = torch.nn.Transformer.generate_square_subsequent_mask(block_size).to(device)
        
        # Atención y suma residual
        attn_output, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), is_causal=True, attn_mask=mask)
        x = x + attn_output
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Apilamos varias capas de razonamiento
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Convertir números a vectores + agregar información de posición
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb 
        
        # Pasar por el cerebro
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ==========================================
# 4. EL ENTRENAMIENTO INTENSIVO
# ==========================================
print(f"\n--- ENTRENANDO EL MINI-GPT ({sum(p.numel() for p in model.parameters())/1e6:.2f} Millones de parámetros) ---")

for iter in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        print(f"Paso {iter:04d} | Error (Loss): {loss.item():.4f}")

print("\n✅ ¡Entrenamiento completado!")

# ==========================================
# 5. LA PRUEBA FINAL: HACERLO HABLAR
# ==========================================
print("\n--- GENERANDO TEXTO ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Empezamos con un token vacío
generado_ids = context

# Hacemos que genere 50 palabras nuevas
for _ in range(50):
    # Recortamos el contexto para no pasarnos del límite
    idx_cond = generado_ids[:, -block_size:]
    logits, _ = model(idx_cond)
    # Enfocarnos solo en el último paso de tiempo
    logits = logits[:, -1, :]
    # Convertir a probabilidades
    probs = F.softmax(logits, dim=-1)
    # Elegir la siguiente palabra
    idx_next = torch.multinomial(probs, num_samples=1)
    # Añadirla a la frase
    generado_ids = torch.cat((generado_ids, idx_next), dim=1)

texto_generado = enc.decode(generado_ids[0].tolist())
print(f"La IA dice:\n\n{texto_generado}")