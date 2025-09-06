import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Doc_reader.doc_up import read_docx
from Doc_reader.pdf_up import read_pdf
from GPT.Gpt_class import Gpt
from Character_level.solution import Solution
#from training_n_pred import gpt_decoder
# ---------- Load & Prepare Text ----------
text = read_pdf("dc machine book.pdf", 17000)

# Build vocabulary
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
print(vocab)
print("Vocab size:", vocab_size)

def encode(s):
    return [stoi[c] for c in s]

# def decode(l):
#     return ''.join([itos[i] for i in l])
def decode(indices):
    # make sure l is list of ints
    return ''.join([itos.get(i, '?') for i in indices])

encoded_text = encode(text)

# ---------- Dataset ----------
block_size = 512  # context length
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

MODEL_PATH = "ch_trained_model.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block_size = 512      # context length
model_dim = 128
num_blocks = 8
num_heads = 8

model = Gpt(
    vocab_size=vocab_size,
    context_length=block_size,
    model_dim=model_dim,
    num_blocks=num_blocks,
    num_heads=num_heads
).to(device)

# -----------------------------
# Load or Train
# -----------------------------
if os.path.exists(MODEL_PATH):
    print(f"✅ Found {MODEL_PATH}, loading trained weights...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print("⚡ No trained model found, starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    EPOCHS = 55
    patience = 5          # how many epochs to wait for improvement
    min_delta = 0.001     # minimum required improvement
    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # Early stopping check with min_delta
        if best_loss - avg_loss >= min_delta:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Loss improved by at least {min_delta}. Model saved.")
        else:
            patience_counter += 1
            print(f"⚠️ No sufficient improvement for {patience_counter} epochs")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered")
                break

    # Save after training
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")



solution = Solution()

question = "what is the impact of AI in healthcare"
start_context = torch.tensor([[stoi[c] for c in question]], dtype=torch.long).to(device)

generated_answer = solution.generate(
    model=model,
    new_chars=300,  # number of characters to generate
    context=start_context,
    context_length=block_size,
    int_to_char=itos,
    temperature=0.8,
    top_k=20
)

print("Q:", question)
print("A:", generated_answer)

# I didn't train model for character-wise generation because it was taking like 1 hour for 2 epoch and its ouput on low training was just gibrish like (dc genfertr rdjks jsheo ldeiocs), but aboce code will work fo that