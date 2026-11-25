from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    boot_low: float
    boot_high: float
    level: float
    n_items: int
    widened: bool
    significant: bool

def _wilson(k: int, n: int, level: float):
    if n<=0 or k<0 or k>n: raise ValueError("require 0 <= k <= n and n > 0")
    # Normal quantile without scipy (Acklam approximation).
    a=(1-level)/2; z=_norm_ppf(1-a); ph=k/n; den=1+z*z/n
    center=(ph+z*z/(2*n))/den; half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)

def _norm_ppf(p):
    # Peter Acklam's rational approximation.
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    if p<.02425:
        q=np.sqrt(-2*np.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p>1-.02425: return -_norm_ppf(1-p)
    q=p-.5; r=q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def wilson_interval(k: int, n: int, level: float=0.95) -> tuple[float,float]:
    if not 0<level<1: raise ValueError("level must be in (0,1)")
    return _wilson(k,n,level)

def _validate(y,p,metric,cls,level,n_resamples):
    y,p=np.asarray(y),np.asarray(p)
    if y.ndim!=1 or p.ndim!=1 or len(y)==0 or y.shape!=p.shape: raise ValueError("inputs must be non-empty equal-length vectors")
    if metric not in ('accuracy','recall'): raise ValueError("unknown metric")
    if metric == 'accuracy' and cls is not None: raise ValueError("cls is only valid for recall")
    if not 0<level<1 or not isinstance(n_resamples,int) or n_resamples<100: raise ValueError("invalid level or n_resamples (<100)")
    if metric=='recall':
        if cls is None: raise ValueError("recall requires cls")
        mask=y==cls
        if not mask.any(): raise ValueError("cls has no true members")
        return (p[mask]==y[mask]).astype(float)
    return (p==y).astype(float)

def _interval(x,level,n_resamples,seed):
    m=len(x); est=float(x.mean()); rng=np.random.default_rng(seed); idx=rng.integers(0,m,size=(n_resamples,m)); stats=x[idx].mean(1)
    alpha=(1-level)/2; blo,bhi=np.quantile(stats,[alpha,1-alpha]); floorlo,floorhi=_wilson(int(x.sum()),m,level)
    lo=min(float(blo),floorlo); hi=max(float(bhi),floorhi)
    return Interval(est,lo,hi,float(blo),float(bhi),level,m,lo<float(blo)-1e-15 or hi>float(bhi)+1e-15,lo>0 or hi<0)

def metric_ci(y_true: np.ndarray,y_pred: np.ndarray,metric: str='accuracy',cls: int|None=None,level: float=.95,n_resamples: int=2000,seed: int=0)->Interval:
    x=_validate(y_true,y_pred,metric,cls,level,n_resamples); r=_interval(x,level,n_resamples,seed)
    return Interval(r.estimate,r.low,r.high,r.boot_low,r.boot_high,r.level,r.n_items,r.widened,False)

def paired_diff_ci(y_true: np.ndarray,pred_a: np.ndarray,pred_b: np.ndarray,metric: str='accuracy',cls: int|None=None,level: float=.95,n_resamples: int=2000,seed: int=0)->Interval:
    a=_validate(y_true,pred_a,metric,cls,level,n_resamples); b=_validate(y_true,pred_b,metric,cls,level,n_resamples)
    if len(a)!=len(b): raise ValueError("paired metric populations differ")
    x=a-b; m=len(x); est=float(x.mean()); rng=np.random.default_rng(seed); idx=rng.integers(0,m,size=(n_resamples,m)); stats=x[idx].mean(1); alpha=(1-level)/2; blo,bhi=np.quantile(stats,[alpha,1-alpha])
    d=int(np.count_nonzero(x)); only_a=int(np.count_nonzero(x>0)); only_b=int(np.count_nonzero(x<0))
    if d:
        ql,qh=_wilson(only_a,d,level); floorlo,floorhi=((2*ql-1)*d/m,(2*qh-1)*d/m)
    else:
        z=_norm_ppf(1-alpha); u=z*z/(m+z*z); floorlo,floorhi=-u,u
    lo=min(float(blo),floorlo); hi=max(float(bhi),floorhi)
    return Interval(est,lo,hi,float(blo),float(bhi),level,m,lo<float(blo)-1e-15 or hi>float(bhi)+1e-15,not (lo<=0<=hi))
