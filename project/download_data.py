"""Download the public BANKING77 dataset; no customer/private data is used."""
from pathlib import Path
import pandas as pd

def main():
    try:
        from datasets import load_dataset
    except ImportError as e: raise SystemExit('Install dependencies first: pip install -r requirements.txt') from e
    ds=load_dataset('PolyAI/banking77')
    out=Path('data'); out.mkdir(exist_ok=True)
    for split in ('train','test'):
        table=ds[split]
        df=table.to_pandas()
        if 'category' not in df.columns and 'label' in df.columns:
            feature=table.features['label']
            if hasattr(feature,'int2str'):
                df['category']=[feature.int2str(int(value)) for value in df['label']]
            else:
                df['category']=df['label'].astype(str)
        if not {'text','category'}.issubset(df.columns):
            raise ValueError(f'Unexpected BANKING77 columns for {split}: {list(df.columns)}')
        df[['text','category']].to_csv(out/f'{split}.csv',index=False)
        print(f'Wrote {len(df)} rows to {out}/{split}.csv')
if __name__=='__main__': main()
