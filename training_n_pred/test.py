# test_pred.py
import os
import string
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from training_n_pred.doc_up import read_docx
from GPT.Gpt_class import Gpt
from training_n_pred.solution import Solution

# ------------------ Config ------------------
DOCX_PATH = "Conversation.docx"
READ_CHARS = 13000           # how many chars read by read_docx (your function param)
CHECKPOINT_PATH = "gpt_checkpoint.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOCK_SIZE = 128            # context window in tokens (words here)
EMBED_DIM = 128
NUM_BLOCKS = 4
NUM_HEADS = 4
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 80                 # adjust as you like
GENERATE_WORDS = 200         # how many words to generate for the sample
SEED_PROMPT = "Benifits of AI"  # your question/seed

# ------------------ Utilities ------------------
def safe_encode_seed(seed_sentence: str, stoi: dict):
    """
    Try to encode a seed sentence robustly for a word-level vocab.
    Returns (encoded_list, missing_tokens).
    """
    tokens = seed_sentence.split()
    encoded = []
    missing = []

    for tok in tokens:
        # candidate normalizations to try
        candidates = [
            tok,
            tok.strip(string.punctuation),
            tok.rstrip(".,;:!?"),
            tok.lower(),
            tok.capitalize()
        ]
        found = False
        for c in candidates:
            if c in stoi:
                encoded.append(stoi[c])
                found = True
                break

        # try punctuation-attached variants (e.g. 'word,' in vocab)
        if not found:
            for suff in [",", ".", ";", ":"]:
                if (tok + suff) in stoi:
                    encoded.append(stoi[tok + suff])
                    found = True
                    break

        if not found:
            missing.append(tok)

    return encoded, missing

# ------------------ Load & Prepare Text ------------------
text = read_docx(DOCX_PATH, READ_CHARS)
words = text.split()
vocab = sorted(list(set(words)))
vocab_size = len(vocab)

print("Vocab size:", vocab_size)
print("Sample vocab (first 80 tokens):", vocab[:80])

stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}

def encode(s: str):
    return [stoi[w] for w in s.split()]

def decode(indices):
    return " ".join([itos[i] for i in indices])

# encode the full text (word-level tokens)
encoded_text = [stoi[w] for w in words]

# ------------------ Dataset ------------------
# Ensure block_size is not larger than available tokens
if len(encoded_text) < 2:
    raise ValueError("Training text too short. Provide more data.")
if len(encoded_text) <= BLOCK_SIZE:
    BLOCK_SIZE = max(1, len(encoded_text) - 1)
    print(f"Warning: Reduced BLOCK_SIZE to {BLOCK_SIZE} due to small dataset.")

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

dataset = TextDataset(encoded_text, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=min(BATCH_SIZE, max(1, len(dataset))), shuffle=True)

# ------------------ Model Setup ------------------
model = Gpt(
    vocab_size=vocab_size,
    context_length=BLOCK_SIZE,
    model_dim=EMBED_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ------------------ Load checkpoint if exists ------------------
if os.path.exists(CHECKPOINT_PATH):
    print("Found checkpoint - loading:", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    # ensure loaded vocab matches; if not, warn (we still proceed)
    if ckpt.get("vocab_size") != vocab_size:
        print("Warning: checkpoint vocab_size != current vocab_size (may cause failures).")
    print("Model loaded from checkpoint.")
else:
    # ------------------ Training ------------------
    print("No checkpoint found — training from scratch.")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)  # (B, T, V)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Save checkpoint (model + vocab metadata)
    torch.save({
        "model_state": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "model_dim": EMBED_DIM,
        "num_blocks": NUM_BLOCKS,
        "num_heads": NUM_HEADS
    }, CHECKPOINT_PATH)
    print("Model saved as", CHECKPOINT_PATH)

# ------------------ Generation ------------------
print("\nSample Generation:")

# Safely encode the seed
seed = SEED_PROMPT
encoded_seed, missing = safe_encode_seed(seed, stoi)
if missing:
    print("Warning: these seed tokens were not found in vocab and were skipped:", missing)

if not encoded_seed:
    # fallback to a token we know exists
    fallback_token = vocab[0]
    print(f"No valid seed tokens found, falling back to '{fallback_token}'")
    encoded_seed = [stoi[fallback_token]]

start_context = torch.tensor([encoded_seed], dtype=torch.long).to(DEVICE)

solution = Solution()
generated_answer = solution.generate(
    model=model,
    new_chars=GENERATE_WORDS,    # number of words to generate
    context=start_context,
    context_length=BLOCK_SIZE,
    int_to_char=itos,
    temperature=0.8,
    top_k=30,
    device=DEVICE,
    join_with=" "               # because this is word-level
)

print("Q:", seed)
print("A:", generated_answer)
