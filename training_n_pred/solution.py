# import torch
# from torch import nn
# from torchtyping import TensorType
# import typing

# #[5].item() -> 5
# class Solution:
#     def generate(self,model,new_chars:int,context:TensorType[int],context_length:int,
#                  int_to_char:dict)->str:
#         generator=torch.manual_seed(0)
#         initial_state=generator.get_state()
#         res=[]
#         #context -> B,T
#         #len(context)=B,len(context.T)=T
#         for i in range(new_chars):
#             if len(context.T)>context_length: #context.T means transpose of context
#                 context = context[:,-context_length:] 

#             prediction = model(context)
#             last_time_step = prediction [:,-1:] #we care for last token predicted by model i-e for final time-step 
#             probs = nn.functional.softmax(last_time_step,dim=-1)#normalize along vocabulary dimension so dim=-1
#             next_char = torch.multinomial(probs,1,generator=generator) #here num_samples=1 
#             generator.set_state(initial_state)
#             context=torch.concat((context,next_char),dim=-1) # B,T -> B,T+1
#             res.append(int_to_char[next_char.item()])

#         return ''.join(res)    

# import torch
# from torch import nn
# from torchtyping import TensorType

# class Solution:
#     def generate(
#         self,
#         model: nn.Module,
#         new_chars: int,
#         context: TensorType[int],
#         context_length: int,
#         int_to_char: dict
#     ) -> str:
#         """
#         Generate text using an autoregressive decoder model.

#         Args:
#             model (nn.Module): Trained model for prediction.
#             new_chars (int): Number of new characters to generate.
#             context (TensorType[int]): Initial context tensor (B, T).
#             context_length (int): Maximum context window length.
#             int_to_char (dict): Mapping from int -> character.

#         Returns:
#             str: Generated text sequence.
#         """

#         # Fix random generator for reproducibility
#         generator = torch.manual_seed(0)
#         init_state = generator.get_state()

#         generated_text = []

#         for _ in range(new_chars):
#             # Keep only last `context_length` tokens
#             if context.size(1) > context_length:
#                 context = context[:, -context_length:]

#             # Forward pass through the model
#             prediction = model(context)  # Shape: (B, T, vocab_size)

#             # Focus on last time step (next-token logits)
#             last_step_logits = prediction[:, -1, :]  # Shape: (B, vocab_size)

#             # Convert logits to probabilities
#             probs = nn.functional.softmax(last_step_logits, dim=-1)

#             # Sample from distribution
#             next_char_idx = torch.multinomial(probs, num_samples=1, generator=generator)

#             # Reset generator state for determinism
#             generator.set_state(init_state)

#             # Append prediction to context
#             context = torch.cat((context, next_char_idx), dim=1)

#             # Map to character and store
#             generated_text.append(int_to_char[next_char_idx.item()])

#         return ''.join(generated_text)

# import torch
# from torch import nn
# from torchtyping import TensorType


# class Solution:
#     def generate(
#         self,
#         model: nn.Module,
#         new_chars: int,
#         context: TensorType[int],
#         context_length: int,
#         int_to_char: dict
#     ) -> str:
#         """
#         Generate text using the trained model.
#         """

#         model.eval()
#         generated_text = []

#         for _ in range(new_chars):
#             if context.size(1) > context_length:
#                 context = context[:, -context_length:]

#             with torch.no_grad():
#                 logits = model(context)  # (B, T, vocab_size)
#                 last_logits = logits[:, -1, :]  # (B, vocab_size)
#                 probs = nn.functional.softmax(last_logits, dim=-1)

#                 next_char_idx = torch.multinomial(probs, num_samples=1)
#                 context = torch.cat([context, next_char_idx], dim=1)

#                 generated_text.append(int_to_char[next_char_idx.item()])

#         return ''.join(generated_text)

# import torch
# from torch import nn
# from torchtyping import TensorType


# class Solution:
#     def generate(
#         self,
#         model: nn.Module,
#         new_chars: int,
#         context: TensorType[int],
#         context_length: int,
#         int_to_char: dict,
#         temperature: float = 1.0,
#         top_k: int = None
#     ) -> str:
#         """
#         Generate text with temperature and optional top-k sampling.
#         """
#         model.eval()
#         generated_text = []

#         for _ in range(new_chars):
#             if context.size(1) > context_length:
#                 context = context[:, -context_length:]

#             with torch.no_grad():
#                 logits = model(context)  # (B, T, vocab_size)
#                 last_logits = logits[:, -1, :]  # (B, vocab_size)

#                 # Apply temperature
#                 logits = last_logits / temperature

#                 # Optionally apply top-k filtering
#                 if top_k is not None:
#                     values, indices = torch.topk(logits, top_k)
#                     mask = torch.full_like(logits, float('-inf'))
#                     mask.scatter_(1, indices, values)
#                     logits = mask

#                 # Convert to probabilities
#                 probs = nn.functional.softmax(logits, dim=-1)

#                 # Sample
#                 next_char_idx = torch.multinomial(probs, num_samples=1)
#                 context = torch.cat([context, next_char_idx], dim=1)

#                 generated_text.append(int_to_char[next_char_idx.item()])

#         return ''.join(generated_text)

import torch
from torch import nn
from torchtyping import TensorType

class Solution:
    def generate(
        self,
        model: nn.Module,
        new_chars: int, # number of characters to generate
        context: TensorType[int], #where to start
        context_length: int,
        int_to_char: dict,
        temperature: float = 1.0,
        top_k: int | None = None,
        device: str = "cpu",
        join_with: str = " "   # default join with space for word-level
    ) -> str:
        model = model.to(device)
        model.eval()
        context = context.to(device)

        generated_tokens = []

        for _ in range(new_chars):
            if context.size(1) > context_length:
                context = context[:, -context_length:]

            with torch.no_grad():
                logits = model(context)                 # (B, T, V)
                last_logits = logits[:, -1, :]          # (B, V)
                logits = last_logits / max(1e-8, temperature)

                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    mask = torch.full_like(logits, float('-inf'))
                    mask.scatter_(1, indices, values)
                    logits = mask

                # Convert to probabilities
                probs = nn.functional.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)   # (B,1)

            # Sample
            context = torch.cat([context, next_idx], dim=1)
            generated_tokens.append(int_to_char[next_idx.item()])

        # If using word-level tokens, join with space. If character-level, set join_with=''
        return join_with.join(generated_tokens)
