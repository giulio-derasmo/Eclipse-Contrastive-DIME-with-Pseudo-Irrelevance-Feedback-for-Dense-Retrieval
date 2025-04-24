from tqdm import tqdm

def load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc="loading collection...."):
            docid, text = line.strip().split("\t")
            collection[docid] = text

    return collection
