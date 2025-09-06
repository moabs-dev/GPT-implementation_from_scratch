# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from Doc_reader.doc_up import read_docx
# from GPT.Gpt_class import Gpt
# from Word_level.solution import Solution
# #from training_n_pred import gpt_decoder
# # ---------- Load & Prepare Text ----------
# text = read_docx("Conversation.docx", 1400)

# # Word-level vocab
# words = text.split()
# vocab = sorted(list(set(words)))
# stoi = {w: i for i, w in enumerate(vocab)}
# itos = {i: w for i, w in enumerate(vocab)}
# vocab_size=len(vocab)
# print(vocab_size)
# print(vocab)
# # def encode(s):  # string -> list of ints
# #     return [stoi[w] for w in s.split()]

# def encode(s):
#     return [stoi.get(c, stoi.get(" ", 0)) for c in s]  # fallback to space or 0

# # def decode(l):  # list of ints -> string
#     return " ".join([itos[i] for i in l])

# encoded_text = encode(text)
# # ---------- Dataset ----------
# block_size = 128  # context length
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

# # ---------- Model ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Gpt(
#     vocab_size=vocab_size,
#     context_length=block_size,
#     model_dim=128,
#     num_blocks=4,
#     num_heads=4
# ).to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# # ---------- Training ----------
# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch, (x, y) in enumerate(dataloader):
#         x, y = x.to(device), y.to(device)
#         logits = model(x)  # (B, T, vocab_size)
#         loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# # Save model
# torch.save(model.state_dict(), "trained_model.pkl")
# print("Model saved as trained_model.pkl")


# # ---------- Text Generation ----------
# print("\nSample Generation:")

# # Start with a newline as initial context
# context = torch.tensor([[stoi[text[0]]]], dtype=torch.long).to(device)

# solution=Solution()

# seed = "AI in healthcare"
# encoded_seed = [stoi[w] for w in seed.split() if w in stoi]
# start_context = torch.tensor([encoded_seed], dtype=torch.long).to(device)

# generated_answer = solution.generate(
#     model=model,
#     new_chars=300,  # number of characters to generate
#     context=start_context,
#     context_length=block_size,
#     int_to_char=itos,
#     temperature=0.8,
#     top_k=20
# )

# print("Q:", seed)
# print("A:", generated_answer)

# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from Doc_reader.doc_up import read_docx
# from GPT.Gpt_class import Gpt
# from Word_level.solution import Solution

# # ---------- Load & Prepare Text ----------
# text = read_docx("Conversation.docx", 2400)

# # Word-level vocab
# words = text.split()
# vocab = sorted(list(set(words)))
# stoi = {w: i for i, w in enumerate(vocab)}
# itos = {i: w for i, w in enumerate(vocab)}
# vocab_size = len(vocab)

# print("Vocab size:", vocab_size)

# # Encode/decode
# def encode(s):
#     return [stoi.get(w, 0) for w in s.split()]  # unknown words -> 0

# def decode(indices):
#     return " ".join([itos[i] for i in indices])

# encoded_text = encode(text)

# # ---------- Dataset ----------
# block_size = 128
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

# # ---------- Model ----------
# MODEL_PATH = "wrd_trained_model.pkl"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Gpt(
#     vocab_size=vocab_size,
#     context_length=block_size,
#     model_dim=128,
#     num_blocks=4,
#     num_heads=4
# ).to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# # ---------- Training ----------
# if os.path.exists(MODEL_PATH):
#     print(f"✅ Found {MODEL_PATH}, loading trained weights...")
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.eval()
# else:
#     print("⚡ No trained model found, starting training...")
#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
#     criterion = torch.nn.CrossEntropyLoss()

#     epochs = 10
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for batch, (x, y) in enumerate(dataloader):
#             x, y = x.to(device), y.to(device)
#             logits = model(x)  # (B, T, vocab_size)
#             loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

#     # Save model
#     torch.save(model.state_dict(), MODEL_PATH)
#     print(f"Model saved as {MODEL_PATH}")


# solution = Solution()

# seed = "AI in healthcare"
# encoded_seed = [stoi[w] for w in seed.split() if w in stoi]

# # make sure the seed is not empty
# if len(encoded_seed) == 0:
#     encoded_seed = [0]  # fallback to unknown

# start_context = torch.tensor([encoded_seed], dtype=torch.long).to(device)

# generated_answer = solution.generate(
#     model=model,
#     new_chars=100,  # number of words to generate
#     context=start_context,
#     context_length=block_size,
#     int_to_char=itos,
#     temperature=0.8,
#     top_k=20,
#     join_with=" "  # word-level join
# )

# print("Q:", seed)
# print("A:", generated_answer)