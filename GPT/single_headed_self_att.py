import torch
import torch.nn as nn
from torchtyping import TensorType

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


