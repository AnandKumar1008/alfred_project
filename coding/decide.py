import numpy as np
REVIEW = -1

def _check(probs, cost):
    p, c = np.asarray(probs, float), np.asarray(cost, float)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1),1,atol=1e-6): raise ValueError("invalid probability matrix")
    if c.shape != (p.shape[1],p.shape[1]) or not np.isfinite(c).all() or (c<0).any(): raise ValueError("cost must be finite, non-negative and shape (C,C)")
    return p,c

def expected_costs(probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    p,c = _check(probs,cost); return p @ c

def decide(probs: np.ndarray, cost: np.ndarray, threshold: float, max_review: int) -> np.ndarray:
    risks = expected_costs(probs,cost); n=len(risks)
    if np.isnan(threshold) or threshold < 0: raise ValueError("threshold must be finite and non-negative")
    if not isinstance(max_review,(int,np.integer)) or max_review < 0: raise ValueError("max_review must be non-negative integer")
    best=risks.argmin(1); risk=risks[np.arange(n),best]; eligible=np.flatnonzero(risk>=threshold)
    # Highest risk first; stable lower row index tie break.
    chosen=eligible[np.lexsort((eligible,-risk[eligible]))[:max_review]]
    out=best.astype(int); out[chosen]=REVIEW; return out

def realised_cost(decisions: np.ndarray, labels: np.ndarray, cost: np.ndarray, review_cost: float) -> float:
    d,y=np.asarray(decisions,int),np.asarray(labels,int); c=np.asarray(cost,float)
    if d.ndim!=1 or y.shape!=d.shape or len(y)==0 or c.ndim!=2 or c.shape[0]!=c.shape[1] or (y<0).any() or (y>=len(c)).any() or ((d<0)&(d!=REVIEW)).any() or (d>=len(c)).any(): raise ValueError("invalid decisions, labels or cost matrix")
    if not np.isfinite(c).all() or (c<0).any() or not np.isfinite(review_cost) or review_cost<0: raise ValueError("costs must be finite and non-negative")
    return float(np.where(d==REVIEW,review_cost,c[y,np.maximum(d,0)]).mean())

def choose_threshold(probs: np.ndarray, labels: np.ndarray, cost: np.ndarray, review_cost: float, max_review_fraction: float) -> float:
    p,c=_check(probs,cost); y=np.asarray(labels,int); n=len(p)
    if y.shape!=(n,) or not np.issubdtype(y.dtype,np.integer) or (y<0).any() or (y>=p.shape[1]).any(): raise ValueError("invalid labels")
    if not np.isfinite(review_cost) or review_cost<0 or not 0<=max_review_fraction<=1: raise ValueError("invalid review policy")
    budget=int(np.floor(max_review_fraction*n)); risks=expected_costs(p,c); bestq=risks.argmin(1); best_risk=risks[np.arange(n),bestq]
    base=c[y,bestq].astype(float); order=np.lexsort((np.arange(n),-best_risk)); delta=review_cost-base[order]
    prefix=np.r_[0.,np.cumsum(delta)]; base_total=float(base.sum()); best_total=base_total; best_t=float(np.inf)
    # A threshold can only select a prefix ending at a distinct risk boundary;
    # once the review budget is reached, every lower threshold selects the same prefix.
    boundaries=np.unique(best_risk)[::-1]
    for threshold in boundaries:
        m=min(budget,int(np.count_nonzero(best_risk>=threshold)))
        total=(base_total+prefix[m])/n
        if total < best_total-1e-12 or (abs(total-best_total)<=1e-12 and float(threshold)>best_t):
            best_total,best_t=total,float(threshold)
    return best_t
