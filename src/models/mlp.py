from torch import nn

class MLP(nn.Module):
    def __init__(self, dim_layers: list,dtype=float):
        super().__init__()
        self.len_layers = len(dim_layers)
        self.linears = nn.ModuleList([])
        for i in range(self.len_layers - 1):
            self.linears.append(nn.Linear(dim_layers[i],dim_layers[i + 1],dtype=dtype))
            nn.init.xavier_uniform_(self.linears[len(self.linears) - 1].weight)
            nn.init.constant_(self.linears[len(self.linears) -1 ].bias, 0.0) 
    
    def forward(self, features):
        x = self.linears[0](features)
        for linear in self.linears[1:]:
            x = linear(x)
        return x
    