from project.model import fit_model, URGENT_CATEGORIES
from project.data import remove_near_duplicate_training_rows
import pandas as pd

def test_router_smoke_and_urgent_scope():
    texts=[]; labels=[]
    for label,word in [('a','alpha'),('b','beta'),('c','gamma')]:
        for i in range(30): texts.append(f'{word} support question {i}'); labels.append(label)
    router,metrics=fit_model(texts,labels,seed=7)
    result=router.predict('alpha support question')
    assert result['queue'] in {'a','b','c','HUMAN REVIEW'}
    assert len(result['probabilities'])==3 and result['explanation']
    assert metrics['validation_rows']>0
    assert len(URGENT_CATEGORIES)==6

def test_near_duplicate_filter():
    train=pd.DataFrame({'text':['I need help with my card','where is my card','transfer money please'],
                        'category':['a','b','c']})
    kept,audit=remove_near_duplicate_training_rows(train,['I need help with my card'],threshold=.99)
    assert len(kept)==2 and len(audit)==1
