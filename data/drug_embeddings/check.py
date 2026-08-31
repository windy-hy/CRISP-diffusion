import sys

import pandas as pd
import numpy as np

raw = pd.read_parquet('embeddings_lincs_sciplex3.parquet')
rdkit =  pd.read_parquet('fcfp4_1024_embedding_lincs_sciplex3.parquet')

print(1)
