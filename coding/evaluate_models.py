import argparse
import numpy as np
from coding.demo_data import make_demo
from coding.bootstrap import metric_ci, paired_diff_ci

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--n',type=int,default=400); args=ap.parse_args()
    y,a,b=make_demo(args.seed,args.n)
    for name,p in [('A',a),('B',b)]:
        for metric,cls in [('accuracy',None),('recall',2)]:
            r=metric_ci(y,p,metric,cls,seed=args.seed)
            print(f'{name} {metric}: estimate={r.estimate:.3f}, reported=[{r.low:.3f}, {r.high:.3f}], bootstrap=[{r.boot_low:.3f}, {r.boot_high:.3f}], n={r.n_items}, widened={r.widened}')
    for metric,cls in [('accuracy',None),('recall',2)]:
        r=paired_diff_ci(y,a,b,metric,cls,seed=args.seed)
        print(f'A-B {metric}: estimate={r.estimate:.3f}, reported=[{r.low:.3f}, {r.high:.3f}], bootstrap=[{r.boot_low:.3f}, {r.boot_high:.3f}], n={r.n_items}, significant={r.significant}')
if __name__=='__main__': main()
