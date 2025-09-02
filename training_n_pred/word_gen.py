import torch
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

# ---------- Text Generation ----------
start_context = "Hello"
encoded_context = encode(start_context)

solution = Solution()

# generate tokens
generated_ids = solution.generate(
    model=model,
    new_chars=500,                         # how many new characters to generate
    context=torch.tensor(encoded_context, dtype=torch.long).unsqueeze(0).to(device),  
    context_length=len(encoded_context),   # length of the seed text
    int_to_char=itos,               # pass your dictionary
    temperature=1.0,                       # randomness in sampling
    top_k=10,                              # optional (set None if not needed)
    #device=device
)

# decode to text
# generated_text = decode(generated_ids[0].tolist())
print("Generated Text:\n", generated_ids)



# print(solution.generate(model, new_chars=200, context=start_context, 
#                         context_length=128, int_to_char=int_to_char, 
#                         temperature=1.2, top_k=20))



# print("Generated indices:", generated)
# print("Type of elements:", [type(i) for i in generated])

# print(decode(generated))
