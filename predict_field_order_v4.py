"""One saved field-calibration action from observed stock and total open quantity."""
from pathlib import Path
import os,sys,argparse,json
ROOT=Path(__file__).resolve().parent
if os.name=='nt':sys.path.insert(0,str(ROOT/'.vendor'))
import numpy as np
import torch
from inventory_full.models import InventoryNet
from inventory_full.engine import plan
from inventory_full.field_calibration import FieldCalibrator,attach_context,callback

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',default='standard');ap.add_argument('--series',type=int,default=0)
    ap.add_argument('--day',type=int,default=370);ap.add_argument('--stock',type=float,default=50.);ap.add_argument('--open-quantity',type=float,default=30.)
    a=ap.parse_args();out=ROOT/f'results/{a.profile}';cfg=json.loads((out/'config_used.json').read_text('utf-8'));cfg['device']='cpu'
    with np.load(ROOT/f'data/{a.profile}/panel.npz') as z:p={k:z[k] for k in z.files}
    if not 0<=a.series<len(p['scale']) or not 188<=a.day<=p['demand'].shape[1]-cfg['horizon']:ap.error('series/day outside available causal history and planned horizon')
    if a.open_quantity<0:ap.error('open-quantity must be nonnegative')
    ck=torch.load(out/'models/main.pt',map_location='cpu',weights_only=True);s=ck['spec']
    encoder=InventoryNet(H=s['H'],hidden=s['hidden'],layers=s['layers']);encoder.load_state_dict(ck['weights']);encoder.eval()
    ck=torch.load(out/'models/field_matched.pt',map_location='cpu',weights_only=True)
    model=FieldCalibrator();model.load_state_dict(ck['weights']);model.eval()
    P=attach_context(p,plan(p,a.day,1,cfg,anchor=a.day,lead_scale=2),encoder,cfg)
    row=np.flatnonzero(P['ids']==a.series)
    quantity=callback(model,cfg)(P,row,np.array([a.stock]),np.array([a.open_quantity]))
    print(json.dumps(dict(source='SYNTHETIC V4 engineering extension',series=a.series,day=a.day,stock=a.stock,
        observed_open_quantity=a.open_quantity,order_quantity=float(quantity[0]),future_realized_arrival_dates_used=False),indent=2))

if __name__=='__main__':main()
