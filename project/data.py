from pathlib import Path
import pandas as pd

def load_banking77(data_dir='data'):
    root=Path(data_dir); train=root/'train.csv'; test=root/'test.csv'
    if not train.exists() or not test.exists(): raise FileNotFoundError('BANKING77 files missing. Run: python -m project.download_data')
    tr=pd.read_csv(train); te=pd.read_csv(test)
    for name,df in [('train',tr),('test',te)]:
        if not {'text','category'}.issubset(df.columns): raise ValueError(f'{name}.csv must contain text and category columns')
        if df[['text','category']].isna().any().any(): raise ValueError(f'{name}.csv contains missing text/category')
    return tr,te

def remove_near_duplicate_training_rows(train_df, test_texts, threshold=0.96):
    """Remove train rows whose character-ngram cosine similarity to any test text >= threshold.

    Returns the filtered frame and an audit frame. Only text similarity is used; test labels
    are never consulted. A high threshold intentionally favors precision over recall.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    if not 0 < threshold <= 1: raise ValueError('threshold must be in (0, 1]')
    if len(train_df)==0 or len(test_texts)==0: raise ValueError('train and test text collections must be non-empty')
    train_texts=train_df.text.astype(str).tolist()
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,max_features=50000,sublinear_tf=True)
    train_x=vectorizer.fit_transform(train_texts); test_x=vectorizer.transform(list(map(str,test_texts)))
    nn=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=-1).fit(test_x)
    distances,indices=nn.kneighbors(train_x)
    similarity=1.0-distances[:,0]; matched=similarity>=threshold
    audit=[]
    for i in np.flatnonzero(matched):
        audit.append({'train_text':train_texts[i],'nearest_test_text':str(test_texts[int(indices[i,0])]),
                      'char_ngram_cosine_similarity':float(similarity[i]),'removed':True})
    return train_df.loc[~matched].reset_index(drop=True),pd.DataFrame(audit)
