#!/usr/bin/env python3
"""Generate parameter table for ML configuration saving runs."""
import argparse,csv
from pathlib import Path

def temps():
    vals=[round(0.6+0.1*i,10) for i in range(17)]
    vals += [round(1.05+0.05*i,10) for i in range(12)]
    return sorted(set(vals))
def label(T,low,high):
    if T<=low: return 1
    if T>=high: return 0
    return -1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outfile',default='parameters/ml_config_scan_D1_L16.csv'); ap.add_argument('--L',type=int,default=16); ap.add_argument('--seeds',default='1,2,3,4,5'); ap.add_argument('--J',type=float,default=1.0); ap.add_argument('--D',type=float,default=1.0); ap.add_argument('--therm',type=int,default=2000); ap.add_argument('--samples',type=int,default=2500); ap.add_argument('--skip',type=int,default=5); ap.add_argument('--proposal',type=float,default=0.7); ap.add_argument('--save-every',type=int,default=5); ap.add_argument('--max-configs',type=int,default=250); ap.add_argument('--low-max',type=float,default=0.9); ap.add_argument('--high-min',type=float,default=1.8); ap.add_argument('--config-out',default='results/configs_ml'); args=ap.parse_args()
    out=Path(args.outfile); out.parent.mkdir(parents=True,exist_ok=True); seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    fields=['run_id','L','T','J','D','Bx','By','Bz','seed','therm','samples','skip','proposal','init','save_every','max_configs','config_out','label']
    rid=0
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for T in temps():
            lab=label(T,args.low_max,args.high_min)
            for seed in seeds:
                w.writerow({'run_id':rid,'L':args.L,'T':T,'J':args.J,'D':args.D,'Bx':0.0,'By':0.0,'Bz':0.0,'seed':seed,'therm':args.therm,'samples':args.samples,'skip':args.skip,'proposal':args.proposal,'init':'helical_z','save_every':args.save_every,'max_configs':args.max_configs,'config_out':args.config_out,'label':lab}); rid+=1
    print(f'Wrote {rid} ML config runs to {out}'); print(f'sbatch --array=0-{rid-1}%64 scripts/submit_ml_config_array.sh')
if __name__=='__main__': main()
