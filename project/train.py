import argparse, json, joblib
import pandas as pd
from pathlib import Path
from project.data import load_banking77, remove_near_duplicate_training_rows
from project.model import fit_model, evaluate, normalize
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data'); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    train,test=load_banking77(args.data_dir)
    # Exclude exact normalized text overlaps from the training pool so test duplicates cannot leak.
    test_texts=test.text.astype(str).tolist(); test_norm={normalize(x) for x in test_texts}
    train_norm=train.text.astype(str).map(normalize)
    overlap_mask=train_norm.isin(test_norm)
    removed=int(overlap_mask.sum()); train=train.loc[~overlap_mask].reset_index(drop=True)
    train,near_audit=remove_near_duplicate_training_rows(train,test_texts,threshold=.96)
    near_removed=len(near_audit)
    router,val=fit_model(train.text.astype(str).tolist(),train.category.astype(str).tolist(),args.seed)
    result=evaluate(router,test_texts,test.category.astype(str).tolist())
    test_probs=router.predict_proba(router.vectorizer.transform(test_texts))
    test_pred=router.encoder.inverse_transform(test_probs.argmax(axis=1))
    urgent_set=__import__('project.model',fromlist=['URGENT_CATEGORIES']).URGENT_CATEGORIES
    errors=[]
    for i,(truth,predicted) in enumerate(zip(test.category.astype(str).tolist(),test_pred)):
        if truth != predicted:
            errors.append({'text':test_texts[i],'true_category':truth,'predicted_category':str(predicted),
                           'confidence':float(test_probs[i].max()),
                           'urgent_false_negative':bool(truth in urgent_set and predicted not in urgent_set)})
    errors.sort(key=lambda row:(not row['urgent_false_negative'],-row['confidence']))
    Path('reports').mkdir(exist_ok=True)
    pd.DataFrame(errors).to_csv('reports/error_audit.csv',index=False)
    near_audit.to_csv('reports/near_duplicate_audit.csv',index=False)
    dummy=DummyClassifier(strategy='most_frequent').fit(train.text,train.category)
    dummy_pred=dummy.predict(test_texts)
    majority_baseline={'accuracy':float(accuracy_score(test.category,dummy_pred)),
                       'macro_f1':float(f1_score(test.category,dummy_pred,average='macro'))}
    Path('models').mkdir(exist_ok=True); Path('reports').mkdir(exist_ok=True)
    joblib.dump(router,'models/router.joblib')
    payload={'exact_train_test_overlap_rows_removed':removed,'near_duplicate_train_rows_removed':near_removed,'near_duplicate_similarity_threshold':0.96,'majority_baseline':majority_baseline,'validation':val,'test':{k:v for k,v in result.items() if k!='report'},'test_classification_report':result['report'],
             'classes':list(router.encoder.classes_),'urgent_categories':sorted(__import__('project.model',fromlist=['URGENT_CATEGORIES']).URGENT_CATEGORIES)}
    Path('reports/evaluation.json').write_text(json.dumps(payload,indent=2))
    template=Path('project/report_template.md').read_text()
    v=val; t=result
    summary=(f"\n## Measured results (seed={args.seed})\n\n"
             f"| Split | Accuracy | Macro-F1 | ECE | Urgent recall | Review rate | Mean cost |\n|---|---:|---:|---:|---:|---:|---:|\n"
             f"| Calibration/threshold validation | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | {v['ece']:.4f} | {v['urgent_recall']:.4f} | {v['review_fraction']:.4f} | {v['mean_realised_cost']:.4f} |\n"
             f"| BANKING77 test | {t['accuracy']:.4f} [{t['accuracy_ci'][0]:.4f}, {t['accuracy_ci'][1]:.4f}] | {t['macro_f1']:.4f} | {t['ece']:.4f} | {t['urgent_seen_recall_including_review']:.4f} | {t['review_fraction']:.4f} | {t['mean_realised_cost']:.4f} |\n\n"
             f"Temperature: {router.temperature:.4f}. Review threshold: {router.threshold}.\n"
             f"Exact normalized overlaps removed: {removed}; near-duplicate train rows removed at character-ngram cosine >= 0.96: {near_removed}. See reports/near_duplicate_audit.csv. Majority baseline: accuracy={majority_baseline['accuracy']:.4f}, macro-F1={majority_baseline['macro_f1']:.4f}.\n"
             f"Test urgent-recall percentile/Wilson-floor interval: {t['urgent_recall_ci']}.\n"
             f"Batch vectorization + model inference: {t['inference_ms_per_message']:.3f} ms/message (test batch average; single-message latency may differ).\n"
             f"Misclassified test examples: {len(errors)}; urgent false negatives: {sum(row['urgent_false_negative'] for row in errors)}. Full examples are in reports/error_audit.csv.\n")
    Path('reports/REPORT.md').write_text(template + summary)
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
