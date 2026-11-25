import numpy as np

def make_demo(seed: int=0,n: int=400):
    rng=np.random.default_rng(seed); y=rng.choice(3,size=n,p=[.70,.25,.05])
    def model(acc_per_class):
        right=rng.random(n)<np.asarray(acc_per_class)[y]
        return np.where(right,y,(y+rng.integers(1,3,size=n))%3)
    return y,model([.92,.80,.70]),model([.90,.85,.55])
