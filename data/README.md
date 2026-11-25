# Dataset setup

The assignment specifies the public BANKING77 dataset by PolyAI, licensed CC BY 4.0. Dataset files are intentionally not bundled in this ZIP. Run `python -m project.download_data` from the repository root to create `data/train.csv` and `data/test.csv` with `text` and `category` columns. The downloader handles the Hugging Face `label`/ClassLabel schema and converts labels to category names.

Do not put private customer messages in this directory or commit downloaded data to a public repository.
