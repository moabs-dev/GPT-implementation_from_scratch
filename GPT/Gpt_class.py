# import torch
# from torch import nn
# from torchtyping import TensorType
# import typing

# class Gpt(nn.Module):
#     def __init__(self,vocab_size:int,context_length:int,model_dim:int,num_blocks:int,num_heads:int):
#         super().__init__()
#         torch.manual_seed(0)
#         self.token_embeddings=nn.Embedding(vocab_size,model_dim)
#         self.pos_embeddings=nn.Embedding(context_length,model_dim)
#         self.blocks=nn.Sequential()
#         for i in range(num_blocks):
#             self.blocks.append(self.TransformerBlock(model_dim,num_heads))

#         self.final_ln=nn.LayerNorm(model_dim)
#         # B,T,D -> B,T,D 
#         self.vocab_projection=nn.Linear(model_dim,vocab_size)# input is model_dim & output is vocab_size 

#     def forward(self,context:TensorType[float])->TensorType[float]:
#         torch.manual_seed(0)
#         token_embeds=self.token_embeddings(context) #B,T,D
#         B,T,D =token_embeds.shape
#         pos_embeds=self.pos_embeddings(torch.arange(T))
#         total_embeddings= token_embeds+pos_embeds

#         un_normalized = self.vocab_projection(self.final_ln(self.blocks(total_embeddings)))
#         probs = nn.functional.softmax(un_normalized,dim=-1)

#         return torch.round(probs,decimals=4)

#     class TransformerBlock(nn.Module):

#         class MultiHeadedAttention(nn.Module):
        
#             class SingleHeadAttention(nn.Module):
        
#                 def __init__(self, embedding_dim: int, attention_dim: int):
#                     super().__init__()
#                     torch.manual_seed(0)
#                     self.get_keys = nn.Linear(embedding_dim,attention_dim,bias=False)
#                     self.get_querries = nn.Linear(embedding_dim,attention_dim,bias=False)
#                     self.get_values = nn.Linear(embedding_dim,attention_dim,bias=False)
                    
#                 def forward(self, embedded: TensorType[float]) -> TensorType[float]: 
                                    
#                     k=self.get_keys(embedded)
#                     q=self.get_querries(embedded)
#                     v=self.get_values(embedded)
               
#                     # '@' is for matmul here  
            
#                     scores= q @ torch.transpose(k,1,2)
#                     B,T,A = k.shape
#                     scores = scores /(A ** 0.5)
            
#                     pre_mask= torch.tril(torch.ones(T,T))
#                     mask = pre_mask == 0
                    
#                     scores = scores.masked_fill(mask,float('-inf')) 
#                     scores = nn.functional.softmax(scores,dim=2)
    
#                     transformed_output = scores @ v
        
#                     return torch.round(transformed_output,decimals=4)
            
        
#             def __init__(self,embedding_dim:int ,attention_dim:int,num_heads:int):
#                 super().__init__()
#                 torch.manual_seed(0)
                
#                 self.heads=nn.ModuleList()
#                 for i in range(num_heads):
#                     head_dim = embedding_dim // num_heads
#                     self.heads.append(self.SingleHeadAttention(embedding_dim,head_dim))
        
#                  # Final projection back to embedding_dim
#                 self.proj = nn.Linear(attention_dim, embedding_dim)
            
#             def forward(self,embedded):
#                 #Return answer to 4 decimal places
#                 outputs=[]
#                 for head in self.heads:
#                     outputs.append(head(embedded))
        
#                 cated=torch.cat(outputs,dim=2)#Concatinate dim=2(which is attention_dim) in B,T,A
#                 projected=self.proj(cated)
#                 return torch.round(projected,decimals=4)   
        
#         def __init__(self,model_dim:int,num_heads:int):
#             super().__init__()
#             torch.manual_seed(0)
#             # self.mhsa=self.MultiHeadedAttention(model_dim,model_dim,num_heads)
#             self.mhsa=self.MultiHeadedAttention(model_dim,model_dim,num_heads)
#             self.first_ln=nn.LayerNorm(model_dim)
#             self.second_ln=nn.LayerNorm(model_dim)
#             self.ff=self.VanillaNeuralNetwork(model_dim)
    
#         class VanillaNeuralNetwork(nn.Module):
#             def __init__(self, model_dim:int):
#                 super().__init__()
#                 torch.manual_seed(0)
#                 self.one=nn.Linear(model_dim,model_dim)
#                 self.two=nn.Linear(model_dim,model_dim)
    
#             def forward(self,now:TensorType[float])->TensorType[float]:
#                 torch.manual_seed(0)
#                 first=self.one(now)
#                 final=nn.functional.relu(self.two(first))
#                 return torch.round(final,decimals=4)    

#         def forward(self,embedded : TensorType[float]) ->TensorType[float]:
#             torch.manual_seed(0)
#             first_part=embedded+self.mhsa(self.first_ln(embedded))
#             res = first_part +self.ff(self.second_ln(first_part))
#             return torch.round(res,decimals=4)
    
    
    
        
import torch
from torch import nn
from torchtyping import TensorType

#Gpt-3 small has: num_blocks=12,num_heads=12,context_length=1048,model_dim=768
class Gpt(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.pos_embeddings = nn.Embedding(context_length, model_dim)
        self.blocks=nn.Sequential()
        for i in range(num_blocks):
            self.blocks.append(self.TransformerBlock(model_dim,num_heads))
        # self.blocks = nn.Sequential(*[
        #     self.TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)
        # ])

        self.final_ln = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)  # logits output

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        token_embeds = self.token_embeddings(context)  # (B, T, D)
        B, T, D = token_embeds.shape
        pos_embeds = self.pos_embeddings(torch.arange(T, device=context.device))
        total_embeddings = token_embeds + pos_embeds

        x = self.blocks(total_embeddings)
        x = self.final_ln(x)
        logits = self.vocab_projection(x)  # (B, T, vocab_size)
        return logits  # keep as raw logits (no softmax!)

    class TransformerBlock(nn.Module):
        class MultiHeadedAttention(nn.Module):
            class SingleHeadAttention(nn.Module):
                def __init__(self, embedding_dim: int, attention_dim: int):
                    super().__init__()
                    self.get_keys = nn.Linear(embedding_dim, attention_dim, bias=False)
                    self.get_queries = nn.Linear(embedding_dim, attention_dim, bias=False)
                    self.get_values = nn.Linear(embedding_dim, attention_dim, bias=False)

                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    k = self.get_keys(embedded)
                    q = self.get_queries(embedded)
                    v = self.get_values(embedded)

                    scores = q @ torch.transpose(k, 1, 2)
                    B, T, A = k.shape
                    scores = scores / (A ** 0.5)

                    mask = torch.tril(torch.ones(T, T, device=embedded.device)) == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim=-1)

                    return scores @ v

            def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
                super().__init__()
                self.heads = nn.ModuleList([
                    self.SingleHeadAttention(embedding_dim, embedding_dim // num_heads)
                    for _ in range(num_heads)
                ])
                self.proj = nn.Linear(embedding_dim, embedding_dim)

            def forward(self, embedded):
                outputs = [head(embedded) for head in self.heads]
                cated = torch.cat(outputs, dim=-1)
                return self.proj(cated)

        class VanillaNeuralNetwork(nn.Module):
            def __init__(self, model_dim: int):
                super().__init__()
                self.one = nn.Linear(model_dim, model_dim)
                self.two = nn.Linear(model_dim, model_dim)

            def forward(self, now: TensorType[float]) -> TensorType[float]:
                return nn.functional.relu(self.two(self.one(now)))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            self.mhsa = self.MultiHeadedAttention(model_dim, model_dim, num_heads)
            self.first_ln = nn.LayerNorm(model_dim)
            self.second_ln = nn.LayerNorm(model_dim)
            self.ff = self.VanillaNeuralNetwork(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            x = embedded + self.mhsa(self.first_ln(embedded))
            x = x + self.ff(self.second_ln(x))
            return x
        
        
    
        
    
    
