"""Predict a nonnegative replenishment action from the saved network and known inputs."""
from pathlib import Path
import argparse,os,sys,json
ROOT=Path(__file__).resolve().parent
if os.name=='nt':sys.path.insert(0,str(ROOT/'.vendor'))
import numpy as np
import torch
from inventory_full.data import features
from inventory_full.models import InventoryNet

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',default='standard')
    ap.add_argument('--series',type=int,default=0);ap.add_argument('--day',type=int,default=330)
    ap.add_argument('--stock',type=float,default=50.);args=ap.parse_args()
    with np.load(ROOT/f'data/{args.profile}/panel.npz') as z:p={k:z[k] for k in z.files}
    ck=torch.load(ROOT/f'results/{args.profile}/models/main.pt',map_location='cpu',weights_only=True)
    spec=ck['spec'];H=spec['H']
    if not 0<=args.series<len(p['scale']):ap.error('series ID outside panel')
    if not 120<=args.day<=p['demand'].shape[1]-H:ap.error('day must permit historical and known-future features')
    x=features(p,[args.series],[args.day],[args.stock],H)
    net=InventoryNet(H=H,hidden=spec['hidden'],layers=spec['layers']);net.load_state_dict(ck['weights']);net.eval()
    with torch.no_grad():q=float(net({k:torch.as_tensor(v,dtype=torch.float32) for k,v in x.items()},forking=False)['order'][0])*float(p['scale'][args.series])
    print(json.dumps({'source':'SYNTHETIC example','series_id':args.series,'day':args.day,'stock_input':args.stock,
        'order_quantity':q,'nonnegative':q>=0,'not_using_actual_future_demand_or_lead':True},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
