"""MQ-RNN-style LSTM + global/local decoder + shared forking heads.
Reference: Wen et al., arXiv:1711.11053v2 sec3. Specific widths/losses are declared reconstruction choices.
"""
import torch
from torch import nn
from torch.nn import functional as F

class InventoryNet(nn.Module):
    def __init__(self,H=20,hidden=50,layers=2,context=16,dropout=.2):
        super().__init__();self.H=H;self.context=context;self.hidden=hidden
        self.encoder=nn.LSTM(3+12,hidden,batch_first=True)
        self.global_decoder=nn.Sequential(nn.Linear(hidden+12+H*3,128),nn.ReLU(),nn.Linear(128,(H+1)*context))
        self.local_decoder=nn.Sequential(nn.Linear(context*2+3,64),nn.ReLU())
        self.daily_quantiles=nn.Linear(64,7);self.daily_mean=nn.Linear(64,1)
        self.total_quantiles=nn.Linear(64,7)
        self.vlt=nn.Sequential(nn.Linear(24,50),nn.ReLU(),nn.Dropout(dropout),nn.Linear(50,20),nn.ReLU())
        self.vlt_head=nn.Linear(20,1)
        dim=hidden+context+22+20+1
        pre=[]
        for _ in range(layers-1):pre.extend([nn.Linear(dim,100),nn.ReLU(),nn.Dropout(dropout)]);dim=100
        self.pre_stock=nn.Sequential(*pre) if pre else nn.Identity()
        self.with_stock=nn.Sequential(nn.Linear(dim+1,100),nn.ReLU(),nn.Dropout(dropout),nn.Linear(100,1))
        for name,p in self.named_parameters():
            if 'weight' in name:nn.init.normal_(p,0,.01)
            else:nn.init.zeros_(p)
        import numpy as np
        increments=np.diff([0,.3,1.,1.15,1.3,1.5,1.8,2.1])
        for layer in [self.daily_quantiles,self.total_quantiles]:
            layer.bias.data.copy_(torch.tensor(np.log(np.expm1(increments)),dtype=torch.float32))
        self.with_stock[-1].bias.data.fill_(3.)

    def forward(self,x,forking=True):
        static=x['static'];seq=x['seq']
        states,_=self.encoder(torch.cat([seq,static[:,None].expand(-1,seq.shape[1],-1)],dim=-1))
        hs=states[:,[19,23,27]] if forking else states[:,-1:]
        known=x['known'] if forking else x['known'][:,-1:]
        B,K=hs.shape[:2]
        g=self.global_decoder(torch.cat([hs,static[:,None].expand(-1,K,-1),known.flatten(2)],dim=-1))
        g=g.reshape(B,K,self.H+1,self.context)
        local=self.local_decoder(torch.cat([g[:,:,:self.H],g[:,:,-1:,].expand(-1,-1,self.H,-1),known],dim=-1))
        daily=torch.cumsum(F.softplus(self.daily_quantiles(local)),dim=-1)
        mean=F.softplus(self.daily_mean(local)).squeeze(-1)
        totals=torch.cumsum(F.softplus(self.total_quantiles(local)),dim=-1)*torch.arange(1,self.H+1,device=seq.device)[None,None,:,None]
        lh=self.vlt(torch.cat([x['vl'],static],dim=-1))
        lt=F.softplus(self.vlt_head(lh)).squeeze(-1)+.25
        rep=self.pre_stock(torch.cat([hs[:,-1],g[:,-1,-1],x['stats'],lh,x['review'][:,None]],dim=-1))
        order=self.order_from_rep(rep,x['stock'])
        return dict(order=order,daily=daily,mean=mean,total=totals,lead=lt,rep=rep)

    def order_from_rep(self,rep,stock):
        return F.softplus(self.with_stock(torch.cat([rep,stock[:,None]],dim=-1))).squeeze(-1)
