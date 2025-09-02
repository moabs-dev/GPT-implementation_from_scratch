# solution.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
from GPT.Gpt_class import Gpt  # your existing GPT model
from training_n_pred.doc_up import read_docx
# ---------------------------
# 1. Tokenization (word-level)
# ---------------------------
def tokenize(text):
    # simple word-level tokenizer (split on words, keep punctuation)
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens



text = read_docx("Conversation.docx", 1400)

tokens = tokenize(text)

# Build vocab
word_counts = Counter(tokens)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_int = {word: i for i, word in enumerate(vocab)}
int_to_word = {i: word for i, word in enumerate(vocab)}

vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# Encode tokens → ints
encoded = [word_to_int[w] for w in tokens]

# ---------------------------
# 2. Dataset preparation
# ---------------------------
seq_length = 20  # words per training sequence

class WordDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

dataset = WordDataset(encoded, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------------------------
# 3. Model + Training
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
`model = Gpt(vocab_size=vocab_size, context_length=seq_length, num_blocks=4, num_heads=4, model_dim=128).to(device)'
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

epochs = 500
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        logits = model(x)   # forward only takes x
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Save model + vocab
torch.save(model.state_dict(), "gpt_wordlevel.pth")
np.save("word_to_int.npy", word_to_int)
np.save("int_to_word.npy", int_to_word)
