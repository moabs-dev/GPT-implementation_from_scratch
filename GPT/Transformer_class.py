import torch
from torch import nn
import torchtyping
from torchtyping import TensorType
import typing

class TransformerBlock(nn.Module):
    def __init__(self,model_dim:int,num_heads:int):
        super.__init__()
        torch.manual_seed(0) #below, att_dim=embd_dim=model_dim
        self.mhsa=self.MultiHeadedAttention(model_dim,model_dim,num_heads)#from contructor of this class MultiHeadedAttention
        self.first_ln=nn.LayerNorm(model_dim) #for part 1 Transblock
        self.second_ln=nn.LayerNorm(model_dim) #for part 2 Transblock
        self.ff=self.VanillaNeuralNetwork(model_dim) 

        
    def forward(self,embedded : TensorType[float]) ->TensorType[float]:
        torch.manual_seed(0)
        first_part=embedded+self.mhsa(self.first_ln(embedded))
        res = first_part +self.ff(self.second_ln(first_part))
        return torch.round(res,decimals=4)

    class VanillaNeuralNetwork(nn.Module):
        def __init__(self, model_dim:int):
            super().__init__()
            torch.manual_seed(0)
            self.one=nn.Linear(model_dim,model_dim)
            self.two=nn.Linear(model_dim,model_dim)

        def forward(self,now:TensorType[float])->TensorType[float]:
            torch.manual_seed(0)
            first=self.one(now)
            final=nn.functional.relu(self.two(first))
            return torch.round(final,decimals=4)


    class MultiHeadedAttention(nn.Module):
    
        class SingleHeadAttention(nn.Module):
    
            def __init__(self, embedding_dim: int, attention_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.get_keys = nn.Linear(embedding_dim,attention_dim,bias=False)
                self.get_querries = nn.Linear(embedding_dim,attention_dim,bias=False)
                self.get_values = nn.Linear(embedding_dim,attention_dim,bias=False)
                
            def forward(self, embedded: TensorType[float]) -> TensorType[float]: 
                                
                k=self.get_keys(embedded)
                q=self.get_querries(embedded)
                v=self.get_values(embedded)
           
                # '@' is for matmul here  
        
                scores= q @ torch.transpose(k,1,2)
                B,T,A = k.shape
                scores = scores /(A ** 0.5)
        
                pre_mask= torch.tril(torch.ones(T,T))
                mask = pre_mask == 0
                
                scores = scores.masked_fill(mask,float('-inf')) 
                scores = nn.functional.softmax(scores,dim=2)

                transformed_output = scores @ v
    
                return torch.round(transformed_output,decimals=4)
        
    
        def __init__(self,embedding_dim:int ,attention_dim:int,num_heads:int):
            super().__init__()
            torch.manual_seed(0)
            
            self.heads=nn.ModuleList()
            for i in range(num_heads):
                self.heads.append(self.SingleHeadedAttention(embedding_dim,attention_dim//num_heads))
    
        def forward(self,embedded):
            #Return answer to 4 decimal places
            outputs=[]
            for head in self.heads:
                outputs.append(head(embedded))
    
            cated=torch.cat(outputs,dim=2)#Concatinate dim=2(which is attention_dim) in B,T,A
            return torch.round(cated,decimals=4)   
    
    
    

    


