import os
import random
import logging
import torch
import tiktoken
from datasets import load_dataset
from src.models.model import MiniGPT 

# 1. CONFIGURACIÓN E INGENIERÍA DE LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IA_NAME = "Hannah"
BLOCK_SIZE = 256  
BATCH_SIZE = 128 # Reducido ligeramente para asegurar estabilidad de memoria
LEARNING_RATE = 3e-5
TOTAL_STEPS = 10000

BASE_MODEL_PATH = 'src/models/checkpoints/model_best.pth'
SAVE_PATH = 'src/models/checkpoints/hannah_finetuned.pth'

# 2. CARGA DINÁMICA DEL MODELO
enc = tiktoken.get_encoding("gpt2")
model = MiniGPT(enc.n_vocab, DEVICE).to(DEVICE)

if os.path.exists(BASE_MODEL_PATH):
    checkpoint = torch.load(BASE_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Modelo base cargado. Arquitectura MiniGPT lista.")
else:
    logger.error(f"Error crítico: No existe el modelo base en {BASE_MODEL_PATH}")
    exit(1)

# 3. INGESTA DE DATOS (Manejo de Estructura Anidada)
logger.info("Sincronizando datasets con el cache local...")
try:
    # Cargamos el JSON directamente para evitar el error del script de usuario
    PERSONA_URL = "https://huggingface.co/datasets/bavard/personachat_truecased/resolve/main/personachat_truecased_full_train.json"
    ds_persona = load_dataset("json", data_files=PERSONA_URL, split="train")
    
    # Datasets con cargadores oficiales estables
    ds_daily = load_dataset("roskoN/dailydialog", split="train", trust_remote_code=True)
    ds_empathy = load_dataset("facebook/empathetic_dialogues", split="train", trust_remote_code=True)
except Exception as e:
    logger.error(f"Fallo en la descarga/lectura de datos: {e}")
    exit(1)

# 4. GENERADORES DE ALTO RENDIMIENTO
def stream_persona(ds):
    """Extrae turnos individuales del JSON anidado proporcionado."""
    for conversation in ds:
        persona_txt = " ".join(conversation.get('personality', []))
        for turn in conversation.get('utterances', []):
            history = turn.get('history', [])
            candidates = turn.get('candidates', [])
            if history and candidates:
                # El último candidato es siempre la respuesta real (ground truth)
                yield 'persona', {
                    'persona': persona_txt,
                    'history': history,
                    'response': candidates[-1]
                }

def stream_daily(ds):
    for ex in ds:
        msgs = ex['utterances']
        if len(msgs) >= 2:
            yield 'daily', {'history': msgs[:-1], 'response': msgs[-1]}

def stream_empathy(ds):
    for ex in ds:
        yield 'empathy', {
            'context': ex['context'],
            'history': [ex['utterance'].replace('_comma_', ',')],
            'response': "Entiendo cómo te sientes." # Respuesta genérica de apoyo
        }

def robust_mixed_generator():
    """Garantiza un flujo infinito de datos mezclando fuentes según pesos."""
    sources = {
        'persona': lambda: stream_persona(ds_persona),
        'daily': lambda: stream_daily(ds_daily),
        'empathy': lambda: stream_empathy(ds_empathy)
    }
    # Inicializamos iteradores
    iterators = {k: iter(v()) for k, v in sources.items()}
    weights = [0.5, 0.3, 0.2]
    keys = list(sources.keys())

    while True:
        choice = random.choices(keys, weights=weights)[0]
        try:
            yield next(iterators[choice])
        except StopIteration:
            # Reinicio automático del iterador agotado
            iterators[choice] = iter(sources[choice]())

# 5. FORMATEADOR DE PROMPTS (COHERENCIA DE TEXTO)
def format_example(source, data):
    # Definimos los headers en inglés para mantener la consistencia del tokenizador
    if source == 'persona':
        header = f"Persona: {data['persona']}"
        h, r = data['history'], data['response']
    elif source == 'daily':
        header = f"Context: I am {IA_NAME}, your girlfriend."
        h, r = data['history'], data['response']
    else:
        # Para Empathetic Dialogues
        header = f"Empathy: {data['context']}"
        h, r = data['history'], "I totally understand how you feel." 

    history_str = ""
    for i, msg in enumerate(h[-4:]):
        role = "User" if i % 2 == 0 else IA_NAME
        history_str += f"{role}: {msg} "
    
    # Estructura final: Header | History | Response
    return f"{header} | {history_str.strip()} {IA_NAME}: {r} <|endoftext|>"

# 6. BUCLE DE ENTRENAMIENTO ROBUSTO
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
model.train()
data_gen = robust_mixed_generator()

logger.info(f"Iniciando fine-tuning de {IA_NAME} en {DEVICE}...")

for step in range(1, TOTAL_STEPS + 1):
    batch_x, batch_y = [], []
    
    for _ in range(BATCH_SIZE):
        source, data = next(data_gen)
        full_text = format_example(source, data)
        
        tokens = enc.encode(full_text, allowed_special={'<|endoftext|>'})
        tokens = tokens[:BLOCK_SIZE + 1]
        
        # Padding coherente
        if len(tokens) < BLOCK_SIZE + 1:
            tokens += [50256] * ((BLOCK_SIZE + 1) - len(tokens))
        
        batch_x.append(tokens[:-1])
        # Usamos -100 para que PyTorch CrossEntropy ignore el padding en el cálculo del loss
        batch_y.append([(t if t != 50256 else -100) for t in tokens[1:]])
            
    x_tensor = torch.tensor(batch_x, dtype=torch.long).to(DEVICE)
    y_tensor = torch.tensor(batch_y, dtype=torch.long).to(DEVICE)

    # Autocast para eficiencia (BF16 si está disponible en GPU)
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16 if DEVICE == 'cuda' else torch.float32):
        logits, loss = model(x_tensor, y_tensor)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Clip de gradientes para evitar explosiones en modelos pequeños
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0:
        logger.info(f"Step {step:05d} | Loss: {loss.item():.4f}")

    if step % 2000 == 0:
        checkpoint_path = f"{SAVE_PATH.replace('.pth', '')}_step_{step}.pth"
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
        logger.info(f"Checkpoint guardado: {checkpoint_path}")

# Guardado Final
torch.save({'model_state_dict': model.state_dict()}, SAVE_PATH)
logger.info(f"¡Entrenamiento completado! Hannah está lista en: {SAVE_PATH}")