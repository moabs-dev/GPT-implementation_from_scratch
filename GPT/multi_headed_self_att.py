import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedAttention(nn.Module):
    def __init__(self,embedding_dim:int ,attention_dim:int,num_heads:int):
        super().__init__()
        torch.manual_seed(0)
        #nn.ModuleList() will be important. IT works the same as python list
        #but is useful here since instance variables at any subclass of nn.Module
        #must also be subclass of nn.Module
        # 
        # Use  self.SingleHeadedAttention to instantiate.You have to calculate head_size
        self.heads=nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(self.SingleHeadedAttention(embedding_dim,attention_dim//num_heads))
            #whatever you give in att_dim,lets say, you gave att_dim=128 and gave  num_heads=4,then att_dim for each head will be 128/4=32 and
            #answer of each will be concatenated at the end 

    def forward(self,embedded):
        #Return answer to 4 decimal places
        outputs=[]
        for head in self.heads:
            outputs.append(head(embedded))

        cated=torch.cat(outputs,dim=2)#Concatinate dim=2(which is attention_dim) in B,T,A
        return torch.round(cated,decimals=4)   



class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.get_keys = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.get_querries = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.get_values = nn.Linear(embedding_dim,attention_dim,bias=False)
        # 3 vectors to get from 3 Linear layer and its proven from experimentation that excluding 
        # bias is more better than including bias in Linear layers
        # Z= X1 * W1 + X2 * W2 + X3 * W3 +  ... + Xn * Wn + B(x) <-- exclude bias

    def forward(self, embedded: TensorType[float]) -> TensorType[float]: 
        # input is embedded (tensor containing float values) and function output will also a tensor (float values) 
        
        k=self.get_keys(embedded)
        q=self.get_querries(embedded)
        v=self.get_values(embedded)
   
        # '@' is for matmul here  

        scores= q @ torch.transpose(k,1,2)# only swapping dimension 1(T) and dimension 2(A) vector ,not dim0(B)
        #so scores = BxTxA @ BxAxT  : B will be preserved like its a parallel processing independent thing and output: BxTxT
        B,T,A = k.shape
        scores = scores /(A ** 0.5)

        pre_mask= torch.tril(torch.ones(T,T))
        mask = pre_mask == 0
        #masked fill bcz we wanna get rid of those future tokens for every time step before we apply softmax
        scores = scores.masked_fill(mask,float('-inf')) # give -inf to those future tokens (which were zeroed out) so that after applying softmax, their contribution is zero
        scores = nn.functional.softmax(scores,dim=2)
        # dim=2 bcz Batch won't be included as its our final output,batches don't matter then
        transformed_output = scores @ v

        return torch.round(transformed_output,decimals=4)
    

#Another approach:    

# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, embed_dim, attention_dim, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = attention_dim // num_heads

#         self.query = nn.Linear(embed_dim, attention_dim)
#         self.key = nn.Linear(embed_dim, attention_dim)
#         self.value = nn.Linear(embed_dim, attention_dim)
#         self.out = nn.Linear(attention_dim, embed_dim)

#     def forward(self, x):
#         B, S, _ = x.size()
#         Q = self.query(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
#         K = self.key(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.value(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
#         attn = torch.softmax(scores, dim=-1)
#         context = torch.matmul(attn, V)  # [B, H, S, D]

#         context = context.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
#         return self.out(context)
