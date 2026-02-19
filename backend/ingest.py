import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = Path("data")
JSON_PATH = DATA_DIR / "knowledge_base_from_pdf.json"
XLSX_PATH = DATA_DIR / "MASTER.xlsx"

INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("docs_metadata.pkl")

MODEL = "all-MiniLM-L6-v2"
DIM = 384

def flatten(obj):
    out=[]
    if isinstance(obj,dict):
        for v in obj.values(): out+=flatten(v)
    elif isinstance(obj,list):
        for v in obj: out+=flatten(v)
    elif isinstance(obj,str):
        out.append(obj)
    return out

def main():
    if JSON_PATH.exists():
        raw=json.load(open(JSON_PATH))
        paras=flatten(raw)
    else:
        df=pd.read_excel(XLSX_PATH)
        paras=df.astype(str).agg(" ".join,axis=1).tolist()

    embedder=SentenceTransformer(MODEL)

    docs=[]
    for p in paras:
        docs.append({"text":p,"metadata":{}})

    embs=embedder.encode([d["text"] for d in docs])
    embs=np.array(embs).astype("float32")

    index=faiss.IndexFlatL2(DIM)
    index.add(embs)

    faiss.write_index(index,str(INDEX_PATH))
    pickle.dump(docs,open(META_PATH,"wb"))

    print("DONE")

if __name__=="__main__":
    main()
