from __future__ import annotations
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from coding.calibration import expected_calibration_error, fit_temperature, apply_temperature, reliability_bins
from coding.decide import decide, choose_threshold, realised_cost, REVIEW
from coding.bootstrap import metric_ci

URGENT_CATEGORIES={
 'lost_or_stolen_card','lost_or_stolen_phone','compromised_card',
 'card_payment_not_recognised','cash_withdrawal_not_recognised',
 'direct_debit_payment_not_recognised'}

def normalize(text): return re.sub(r'\W+',' ',str(text).lower()).strip()
def customer_groups(texts):
    # Exact normalized duplicates share a group to prevent leakage across train/validation.
    return np.asarray([normalize(t) for t in texts])

def build_estimator():
    features=FeatureUnion([
      ('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=180000,sublinear_tf=True,strip_accents='unicode')),
      ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=180000,sublinear_tf=True,strip_accents='unicode'))])
    return features, LogisticRegression(C=4.0,max_iter=1200,class_weight='balanced',solver='lbfgs')

def fit_model(texts, labels, seed=42):
    enc=LabelEncoder(); y=enc.fit_transform(np.asarray(labels)); groups=customer_groups(texts)
    # Hold out duplicate groups for honest calibration and threshold selection.
    splitter=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed)
    tr,va=next(splitter.split(texts,y,groups))
    features,base=build_estimator(); X=features.fit_transform(np.asarray(texts)[tr]); base.fit(X,y[tr])
    Xv=features.transform(np.asarray(texts)[va]); logits=base.decision_function(Xv)
    # Fit temperature on held-out logits, separate from the classifier fitting rows.
    # Fit a single scalar temperature on held-out logits; this is transparent and version-independent.
    temperature=fit_temperature(logits,y[va]); Xv2=Xv; final_cal=base
    probs=apply_temperature(logits,temperature); yv=y[va]
    cost=make_cost_matrix(enc.classes_); threshold=choose_threshold(probs,yv,cost,review_cost=3.0,max_review_fraction=.15)
    val_pred=final_cal.predict(Xv2)
    val_decisions=decide(probs,cost,threshold,int(np.floor(.15*len(yv))))
    metrics={'validation_rows':int(len(va)),'accuracy':float(accuracy_score(yv,val_pred)),
             'macro_f1':float(f1_score(yv,final_cal.predict(Xv2),average='macro')),
             'ece':float(expected_calibration_error(probs,yv,n_bins=15)),
             'threshold':float(threshold) if np.isfinite(threshold) else None,'urgent_recall':urgent_recall(enc.inverse_transform(yv),enc.inverse_transform(val_pred)),
             'review_fraction':float(np.mean(val_decisions==REVIEW)),'mean_realised_cost':realised_cost(val_decisions,yv,cost,3.0)}
    return Router(features,final_cal,enc,cost,threshold,temperature),metrics

def make_cost_matrix(classes):
    n=len(classes); c=np.ones((n,n),dtype=float); np.fill_diagonal(c,0.)
    # A missed safety report is materially more expensive than a normal routing error.
    urgent=np.array([x in URGENT_CATEGORIES for x in classes])
    c[:,urgent]=2.
    c[urgent,:]=30.
    c[np.ix_(urgent,urgent)]=1.0
    np.fill_diagonal(c,0.)
    return c

def urgent_recall(y_true,y_pred):
    mask=np.isin(y_true,list(URGENT_CATEGORIES))
    return float(np.mean(np.isin(y_pred[mask],list(URGENT_CATEGORIES)))) if mask.any() else float('nan')

class Router:
    def __init__(self,vectorizer,model,encoder,cost,threshold,temperature=1.0):
        self.vectorizer,self.model,self.encoder,self.cost,self.threshold,self.temperature=vectorizer,model,encoder,cost,threshold,temperature
    def predict_proba(self,x):
        return apply_temperature(self.model.decision_function(x),self.temperature)
    def predict(self,text):
        x=self.vectorizer.transform([text]); p=self.predict_proba(x)[0]; d=decide(p[None,:],self.cost,self.threshold,1)[0]
        # Conservative novelty guard: low confidence or a tiny top-two margin goes to a person.
        order=np.argsort(p); margin=float(p[order[-1]]-p[order[-2]])
        review=bool(d<0 or p.max()<.22 or margin<.04)
        label='HUMAN REVIEW' if review else str(self.encoder.inverse_transform([d])[0])
        return {'queue':label,'review':review,'probabilities':sorted(zip(self.encoder.classes_,p),key=lambda z:-z[1])[:5],
                'confidence':float(p.max()),'margin':margin,'explanation':self.explain(x,int(p.argmax()))}
    def explain(self,x,class_idx,top_n=8):
        # Token-level additive contributions from the logistic-regression feature weights.
        base=self.model
        if not hasattr(base,'coef_'): return []
        names=self.vectorizer.get_feature_names_out(); vals=np.asarray(x.toarray()).ravel()*base.coef_[class_idx]
        inds=np.argsort(vals)[::-1]
        return [(str(names[i]),float(vals[i])) for i in inds if vals[i]>0][:top_n]

def evaluate(router,texts,labels):
    start=time.perf_counter(); x=router.vectorizer.transform(texts); p=router.predict_proba(x)
    inference_ms=1000.0*(time.perf_counter()-start)/max(1,len(texts))
    y=router.encoder.transform(labels); pred=p.argmax(1)
    review_budget=int(np.floor(.15*len(y)))
    decisions=decide(p,router.cost,router.threshold,review_budget)
    true_urgent=np.isin(np.asarray(labels),list(URGENT_CATEGORIES))
    routed_labels=router.encoder.inverse_transform(pred)
    safe_urgent=np.isin(routed_labels,list(URGENT_CATEGORIES)) | (decisions==REVIEW)
    urgent_safe_recall=float(safe_urgent[true_urgent].mean()) if true_urgent.any() else float('nan')
    accuracy_interval=metric_ci(y,pred,metric='accuracy',n_resamples=2000,seed=0)
    urgent_binary_true=true_urgent.astype(int); urgent_binary_pred=np.isin(routed_labels,list(URGENT_CATEGORIES)).astype(int)
    urgent_interval=metric_ci(urgent_binary_true,urgent_binary_pred,metric='recall',cls=1,n_resamples=2000,seed=0) if true_urgent.any() else None
    return {'accuracy':float(accuracy_score(y,pred)),'accuracy_ci':[float(accuracy_interval.low),float(accuracy_interval.high)],
            'urgent_recall_ci':[float(urgent_interval.low),float(urgent_interval.high)] if urgent_interval else None,
            'macro_f1':float(f1_score(y,pred,average='macro')),
            'ece':float(expected_calibration_error(p,y,n_bins=15)),
            'reliability_bins':[b.__dict__ for b in reliability_bins(p,y,n_bins=15)],
            'inference_ms_per_message':float(inference_ms),
            'urgent_recall':urgent_recall(np.asarray(labels),routed_labels),
            'urgent_seen_recall_including_review':urgent_safe_recall,
            'review_fraction':float(np.mean(decisions==REVIEW)),
            'mean_realised_cost':realised_cost(decisions,y,router.cost,review_cost=3.0),
            'report':classification_report(y,pred,labels=np.arange(len(router.encoder.classes_)),target_names=router.encoder.classes_,output_dict=True,zero_division=0)}
