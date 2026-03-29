import os
import sys
import time
import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import tiktoken

# ==========================================
# 0. CONFIGURACIÓN DE RUTAS ABSOLUTAS
# ==========================================
# Calculamos las rutas basándonos en dónde está este archivo (src/models/training.py)
current_dir = os.path.dirname(os.path.abspath(__file__)) # Carpeta 'models'
src_dir = os.path.dirname(current_dir)                   # Carpeta 'src'

# Añadimos 'src' al path de Python para evitar errores de importación
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Ruta exacta donde guardaremos los modelos
checkpoints_dir = os.path.join(current_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)
best_checkpoint_path = os.path.join(checkpoints_dir, "model_best.pth")

# ==========================================
# 1. HIPERPARÁMETROS "ULTRA" (RTX 5070 Ti)
# ==========================================
batch_size = 64       
block_size = 256      
max_iters = 50000     # 🔥 50,000 pasos
eval_interval = 1000  
learning_rate = 3e-4  
min_lr = 1e-5         
eval_iters = 100      
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embd = 384          
n_layer = 6           
n_head = 6            
dropout = 0.2         

print(f"--- Arrancando Motor Serie en: {device.upper()} ---")

# ==========================================
# 2. DATA PIPELINE (Conectando tu corpus local)
# ==========================================
# 🔥 Usamos tu archivo local wikitext.py pasándole la ruta absoluta
wikitext_script = os.path.join(src_dir, "corpus", "wikitext.py")
print(f"📚 Cargando dataset desde script local: {wikitext_script}")
dataset = load_dataset(wikitext_script, "wikitext-103-raw-v1")

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

def encode_data(example):
    return {'ids': enc.encode(example['text'])}

train_data = dataset['train'].map(encode_data, remove_columns=['text'], desc="Tokenizando Train")
val_data = dataset['validation'].map(encode_data, remove_columns=['text'], desc="Tokenizando Val")

train_tensor = torch.tensor([id for fila in train_data for id in fila['ids']], dtype=torch.long)
val_tensor = torch.tensor([id for fila in val_data for id in fila['ids']], dtype=torch.long)

def get_batch(split):
    data = train_tensor if split == 'train' else val_tensor
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==========================================
# 3. LA ARQUITECTURA (Optimizada)
# ==========================================
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, batch_first=True, dropout=dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        T = x.size(1) 
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T).to(device)
        attn_output, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), is_causal=True, attn_mask=mask)
        x = x + attn_output
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=min_lr)

# ==========================================
# 4. GESTIÓN DE MEMORIA (Restauración)
# ==========================================
best_val_loss = float('inf')
start_iter = 0

if os.path.exists(best_checkpoint_path):
    print(f"\n🧠 Cargando el MEJOR modelo desde: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    start_iter = checkpoint.get('iter', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"✅ Memoria restaurada. Retomando desde el paso {start_iter}. Récord actual Val Loss: {best_val_loss:.4f}")
else:
    print(f"\n👶 Iniciando entrenamiento ULTRA desde cero.\nSe guardarán copias en: {checkpoints_dir}")

# ==========================================
# 5. ENTRENAMIENTO PRO
# ==========================================
num_params = sum(p.numel() for p in model.parameters())/1e6
print(f"\n--- ENTRENANDO MODELO DE {num_params:.2f} Millones de parámetros ---")

scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
t0 = time.time()

for iter in range(start_iter, max_iters):
    
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        dt = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        print(f"Paso {iter:05d}/{max_iters} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | LR: {current_lr:.6f} | Tiempo: {dt:.2f}s")
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'iter': iter,
            'best_val_loss': min(losses['val'], best_val_loss)
        }

        # 1. Guardar rastro de migas en la nueva carpeta
        step_filename = os.path.join(checkpoints_dir, f"model_{iter:05d}.pth")
        torch.save(checkpoint_data, step_filename)

        # 2. Auto-limpieza de la nueva carpeta
        todos_los_checkpoints = sorted(glob.glob(os.path.join(checkpoints_dir, "model_[0-9]*.pth")))
        if len(todos_los_checkpoints) > 3:
            viejo = todos_los_checkpoints.pop(0)
            os.remove(viejo)

        # 3. Guardar el récord
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"   🏆 ¡Nuevo récord! Mejor modelo actualizado.")
            
        t0 = time.time()

    xb, yb = get_batch('train')
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    scheduler.step()

print("\n✅ ¡Entrenamiento ULTRA completado!")

# ==========================================
# 6. INFERENCIA
# ==========================================
print("\n--- GENERANDO TEXTO ---")
if os.path.exists(best_checkpoint_path):
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
context = torch.tensor([[1]], dtype=torch.long, device=device) 
generado_ids = context

with torch.no_grad():
    for _ in range(150):
        idx_cond = generado_ids[:, -block_size:]
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        generado_ids = torch.cat((generado_ids, idx_next), dim=1)

texto_generado = enc.decode(generado_ids[0].tolist())
print(f"La IA dice:\n\n{texto_generado}")