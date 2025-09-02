# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import random
# from training_n_pred.doc_up import read_docx
# from GPT.Gpt_class import Gpt
# from GPT.solution import Solution

# text=read_docx("Conversation.docx",400)
# vocab = sorted(list(set(text)))
# vocab_size = len(vocab)
# stoi = {ch: i for i, ch in enumerate(vocab)}
# itos = {i: ch for ch, i in stoi.items()}

# print("Vocab size:", vocab_size)
# def encode(s):
#     return [stoi[c] for c in s]


# def decode(l):
#     return ''.join([itos[i] for i in l])

# # encoded_text = encode(text)
# encoded_text = encode(text)


# block_size = 128 # sequence length

# class TextDataset(Dataset):
#     def __init__(self, data, block_size):
#         self.data = data
#         self.block_size = block_size

#     def __len__(self):
#         return len(self.data) - self.block_size
    
    
#     def __getitem__(self, idx):
#         x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
#         y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
#         return x, y


# dataset = TextDataset(encoded_text, block_size)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Gpt(vocab_size,context_length=500,model_dim=128,num_blocks=4,num_heads=4).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch, (x, y) in enumerate(dataloader):
#         x, y = x.to(device), y.to(device)
#         logits = model(x)
#         loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


# print("\nSample Generation:")
# # Create the context as a proper tensor (batch_size=1, seq_len=1)
# context = torch.tensor([[0]], dtype=torch.long)
# solution=Solution()
# print(solution.generate(model, context_length=200, new_chars=200,context=context,int_to_char={0:'\n',1:' ',2:'!',3:'a',4:'b',5:'c'}))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from training_n_pred.doc_up import read_docx
from GPT.Gpt_class import Gpt
from training_n_pred.solution import Solution
#from training_n_pred import gpt_decoder
# ---------- Load & Prepare Text ----------
text = read_docx("Conversation.docx", 2400)

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

# ---------- Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Gpt(
    vocab_size=vocab_size,
    context_length=block_size,
    model_dim=768,
    num_blocks=12,
    num_heads=12
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ---------- Training ----------
epochs = 10
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for x, y in dataloader:
#         x, y = x.to(device), y.to(device)
#         logits = model(x)
#         loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     avg_loss = total_loss / len(dataloader)
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

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
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# torch.save(model.state_dict(), "gpt_decoder.pth")
# print("Model saved as gpt_decoder.pth")

# model.load_state_dict(torch.load(gpt_decoder, map_location="cpu"))
# model.eval()

# Save model
torch.save(model.state_dict(), "trained_model.pkl")
print("Model saved as trained_model.pkl")


# ---------- Text Generation ----------
print("\nSample Generation:")

# Start with a newline as initial context
context = torch.tensor([[stoi[text[0]]]], dtype=torch.long).to(device)

# solution = Solution()
# generated = solution.generate(
#     model=model,
#     context_length=block_size,
#     new_chars=200,
#     context=context,
#     int_to_char=itos
# )
# ---------- Load Trained Model ----------
# checkpoint = torch.load("gpt_model.pth", map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

solution = Solution()

# ---------- Text Generation ----------
# start_context = "Hello"
# encoded_context = encode(start_context)



# # generate tokens
# generated_ids = solution.generate(
#     model=model,
#     new_chars=500,                         # how many new characters to generate
#     context=torch.tensor(encoded_context, dtype=torch.long).unsqueeze(0).to(device),  
#     context_length=len(encoded_context),   # length of the seed text
#     int_to_char=itos,               # pass your dictionary
#     temperature=1.0,                       # randomness in sampling
#     top_k=10,                              # optional (set None if not needed)
#     #device=device
# )

# # decode to text
# # generated_text = decode(generated_ids[0].tolist())
# print("Generated Text:\n", generated_ids)

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



# print(solution.generate(model, new_chars=200, context=start_context, 
#                         context_length=128, int_to_char=int_to_char, 
#                         temperature=1.2, top_k=20))



# print("Generated indices:", generated)
# print("Type of elements:", [type(i) for i in generated])

# print(decode(generated))
