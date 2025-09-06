import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Doc_reader.doc_up import read_docx
from Doc_reader.pdf_up import read_pdf
from GPT.Gpt_class import Gpt
from Word_level.solution import Solution

# ---------- Load & Prepare Text ----------
text = read_pdf("dc machine book.pdf", 17000)

# Word-level vocab
words = text.split()
vocab = sorted(list(set(words)))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print("Vocab size:", vocab_size)

# Encode/decode
def encode(s):
    return [stoi.get(w, 0) for w in s.split()]  # unknown words -> 0

def decode(indices):
    return " ".join([itos[i] for i in indices])

encoded_text = encode(text)

# ---------- Dataset ----------
block_size = 256
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

dataset = TextDataset(encoded_text, block_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------- Model ----------
MODEL_PATH = "wrd_trained_model.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Gpt(
    vocab_size=vocab_size,
    context_length=block_size,
    model_dim=128,
    num_blocks=8,
    num_heads=8
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ---------- Training ----------
if os.path.exists(MODEL_PATH):
    print(f"✅ Found {MODEL_PATH}, loading trained weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print("⚡ No trained model found, starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 100
    patience = 5
    min_delta = 0.001
    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None   # ✅ initialize here
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
        # Early stopping check
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # ✅ keep best weights in memory
            #print(f"✅ Loss improved to {avg_loss:.4f} (best so far)")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered")
                break
    
    # ---- Save only once at the end ----
    if best_model_state is not None:
        torch.save(best_model_state, MODEL_PATH)
        print(f"✅ Best model saved at end with loss {best_loss:.4f}")
    else:
        print("⚠️ No improvement at all. Model not saved.")
    

# ---------- Text Generation ----------
solution = Solution()

seed =str(input('Enter the starting words:')) #"principle of electric "
encoded_seed = [stoi[w] for w in seed.split() if w in stoi]

# fallback if seed words not in vocab
if len(encoded_seed) == 0:
    encoded_seed = [0]

start_context = torch.tensor([encoded_seed], dtype=torch.long).to(device)

generated_answer = solution.generate(
    model=model,
    new_chars=200,  # number of words to generate
    context=start_context,
    context_length=block_size,
    int_to_char=itos,
    temperature=0.8,
    top_k=20,
    join_with=" "  # word-level join
)

print("Q:", seed)
print("A:", generated_answer)
