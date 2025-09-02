# test_pred.py
import torch
import numpy as np
from GPT.Gpt_class import Gpt

# Load vocab
vocab=word_to_int = np.load("word_to_int.npy", allow_pickle=True).item()
int_to_word = np.load("int_to_word.npy", allow_pickle=True).item()
vocab_size = len(word_to_int)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
# Reload model
seq_length = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# init model with same args you used in training
# Load model
model = Gpt(vocab_size=vocab_size,
            context_length=seq_length,
            num_blocks=4,
            num_heads=4,
            model_dim=128).to(device)

state_dict = torch.load("gpt_wordlevel.pth", map_location=device)
model.load_state_dict(state_dict)

# Switch to evaluation mode (no gradients, dropout disabled)
model.eval()


# ---------------------------
# Generation function
# ---------------------------
# def generate_text(prompt, max_new_tokens=50):
#     tokens = prompt.lower().split()
#     input_ids = [word_to_int.get(t, 0) for t in tokens]  # unknown words → 0
#     input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

#     for _ in range(max_new_tokens):
#         # always feed only the last `seq_length` tokens
#         input_ctx = input_ids[:, -seq_length:]
#         logits = model(input_ctx)
#         logits = logits[:, -1, :]  # last token logits
#         probs = torch.softmax(logits, dim=-1)
#         next_token = torch.multinomial(probs, num_samples=1)
#         input_ids = torch.cat([input_ids, next_token], dim=1)


#     output = " ".join(int_to_word[int(i)] for i in input_ids[0].tolist())
#     return output

def sample_next_token(logits, temperature=1.0, top_k=50):
    logits = logits / temperature
    if top_k is not None:
        values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, indices, values)
        logits = mask
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# def generate_text(prompt, max_new_tokens=50, temperature=0.8, top_k=50):
#     tokens = prompt.lower().split()
#     input_ids = [word_to_int.get(t, 0) for t in tokens]
#     input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

#     for _ in range(max_new_tokens):
#         input_ctx = input_ids[:, -seq_length:]
#         logits = model(input_ctx)
#         logits = logits[:, -1, :]
#         next_token = sample_next_token(logits, temperature, top_k)
#         input_ids = torch.cat([input_ids, next_token], dim=1)

#     output = " ".join(int_to_word[int(i)] for i in input_ids[0].tolist())
#     return output

import torch
import torch.nn.functional as F
#from GPT.Gpt_class import Gpt

# def generate_text(model, start_text, stoi, itos, max_new_tokens=50, temperature=1.0, top_k=None, device="cpu"):
#     """
#     Generate text from a trained GPT model, conditioned on the given start_text.

#     Args:
#         model: trained GPT model
#         start_text: seed text string
#         stoi: dict mapping chars/words -> int
#         itos: dict mapping int -> chars/words
#         max_new_tokens: how many tokens to generate
#         temperature: higher = more random, lower = more greedy
#         top_k: keep only top-k logits for sampling (nucleus-like filtering)
#         device: cpu or cuda
#     """

#     # Encode seed text
#     model.eval()
#     idx = torch.tensor([stoi[ch] for ch in start_text], dtype=torch.long, device=device).unsqueeze(0)

#     for _ in range(max_new_tokens):
#         # Feed *all* context (not just last token!)
#         with torch.no_grad():
#             logits, _ = model(idx)   # (B, T, vocab_size)

#         logits = logits[:, -1, :] / temperature  # only take last step’s logits

#         # Optionally filter logits with top-k
#         if top_k is not None:
#             values, _ = torch.topk(logits, top_k)
#             min_values = values[:, -1].unsqueeze(-1)
#             logits = torch.where(logits < min_values, torch.full_like(logits, -float("Inf")), logits)

#         probs = F.softmax(logits, dim=-1)
#         next_id = torch.multinomial(probs, num_samples=1)

#         # Append sampled id to the sequence
#         idx = torch.cat([idx, next_id], dim=1)

#     # Decode tokens back to string
#     out = ''.join([itos[int(i)] for i in idx[0].tolist()])
#     return out

def generate_text(model, start_text, stoi, itos, block_size, max_new_tokens=100, temperature=0.7, top_k=None):
    model.eval()
    device = next(model.parameters()).device
    
    # Encode seed text into indices
    idx = torch.tensor(
        [stoi[word] for word in start_text.split() if word in stoi],
        dtype=torch.long, device=device
    ).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]   # ✅ use passed block_size
        
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_idx), dim=1)

    # Decode indices back to words
    out = " ".join([itos[i] for i in idx[0].tolist()])
    return out



# ---------------------------
# Test it
# ---------------------------
# prompt = "types of ai"
# print(generate_text(prompt, max_new_tokens=500))

# seed = "types of AI"
# print(generate_text(model, seed, stoi, itos, max_new_tokens=100, temperature=0.7, top_k=50))
generated = generate_text(
    model=Gpt,
    start_text="types of AI",
    stoi=stoi,
    itos=itos,
    block_size=128,   # 👈 whatever you trained with
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)

print(generated)