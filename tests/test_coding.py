import numpy as np
import pytest
from coding.splits import group_stratified_kfold
from coding.calibration import reliability_bins, expected_calibration_error, apply_temperature, fit_temperature
from coding.decide import decide, expected_costs, realised_cost, choose_threshold, REVIEW
from coding.bootstrap import wilson_interval, metric_ci, paired_diff_ci

def test_split_group_exclusive_and_deterministic():
    y=['a','a','b','b']*6; g=[f'g{i//2}' for i in range(24)]
    a=group_stratified_kfold(y,g,k=3,seed=4,tolerance=.5); b=group_stratified_kfold(y,g,k=3,seed=4,tolerance=.5)
    assert [f.test for f in a]==[f.test for f in b]
    for f in a:
        assert set(f.train).isdisjoint(f.test) and sorted(f.train+f.test)==list(range(24))
        assert len({g[i] for i in f.test})==len(set(g[i] for i in f.test))

def test_split_invalid():
    with pytest.raises(ValueError): group_stratified_kfold([],[],2)
    with pytest.raises(ValueError): group_stratified_kfold(['a'],['g'],1)

def test_calibration():
    p=np.array([[.95,.05],[.2,.8],[.6,.4]])
    assert [b.count for b in reliability_bins(p,np.array([0,1,1]),5)]==[0,0,1,1,1]
    assert expected_calibration_error(p,np.array([0,1,1]),5)>=0
    z=np.array([[10000.,0.],[0.,10000.]])
    assert np.isfinite(apply_temperature(z,.05)).all()
    assert 0.05<=fit_temperature(z,np.array([0,1]))<=20

def test_decision():
    p=np.array([[.9,.1],[.2,.8]]); c=np.array([[0.,4.],[2.,0.]])
    assert np.allclose(expected_costs(p,c),[[.2,3.6],[1.6,.8]])
    d=decide(p,c,.5,1); assert list(d)==[0,REVIEW]
    assert realised_cost(d,np.array([0,1]),c,3)==1.5
    assert choose_threshold(p,np.array([0,1]),c,3,.5)>=0

def test_bootstrap_floor_and_paired():
    lo,hi=wilson_interval(20,20); assert lo<1 and hi==1
    y=np.array([0]*10+[1]*10); a=y.copy(); b=np.array([1]*3+[0]*7+[1]*10)
    r=metric_ci(y,a,n_resamples=200,seed=1); assert r.low<=r.estimate<=r.high
    with pytest.raises(ValueError): metric_ci(y,a,'accuracy',cls=1,n_resamples=200)
    d=paired_diff_ci(y,a,b,n_resamples=200,seed=1); assert d.low<=0<=d.high and not d.significant
    r=metric_ci(np.array([0,0]),np.array([0,0]),'recall',0,n_resamples=200); assert r.low<1
