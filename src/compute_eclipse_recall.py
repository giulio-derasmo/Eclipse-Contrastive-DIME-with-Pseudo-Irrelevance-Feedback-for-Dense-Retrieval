import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
sys.path.append(".")

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
import dimension_filters
from memmap_interface import MemmapCorpusEncoding, MemmapQueriesEncoding
from sentence_transformers import SentenceTransformer
import pickle
from copy import deepcopy
from glob import glob
from collections import Counter

def compute_recall(results_df, qrels_df, label_list = None):
    # Get relevant documents
    relevant = qrels_df[qrels_df['relevance'].isin(label_list)]
   
    # Count total relevant docs per query
    relevant_counts = relevant.groupby('query_id').size()
    
    # Find which relevant docs were retrieved
    retrieved_relevant = pd.merge(
        results_df[['query_id', 'doc_id']],
        relevant[['query_id', 'doc_id']],
        on=['query_id', 'doc_id']
    )
    
    # Count retrieved relevant docs per query
    retrieved_counts = retrieved_relevant.groupby('query_id').size()
    # Calculate recall for each query
    recall_by_query = retrieved_counts / relevant_counts
    
    return recall_by_query.mean()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        d = pickle.load(handle)
    print('Total Number of configurations: ', len(d))
    return d

collection2corpus = {"deeplearning19": "msmarco-passage", 
                     "deeplearning20": "msmarco-passage",
                     "deeplearninghd": "msmarco-passage", 
                     "robust04": "robust04",
                     "antique": "antique"}

m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
        "cocondenser": 'sentence-transformers/msmarco-bert-co-condensor'}

FilterToFolder = {"LLMEclipse": "llm", "PRFEclipse": "prf",
                    "TopkFilter": "baselines", "GPTFilter": "baselines", "OracularFilter": "baselines"}

def masked_retrieve_and_evaluate(queries, qrels, qembs, qrys_encoder, mapper, q2r, dim_importance, alpha, index):
    #######
    queries = deepcopy(queries)
    qrels = deepcopy(qrels)
    qembs = deepcopy(qembs)
    mapper = deepcopy(mapper)
    q2r = deepcopy(q2r)
    dim_importance = deepcopy(dim_importance)
    #######

    n_dims = int(np.round(alpha * qrys_encoder.shape[1]))
    selected_dims = dim_importance.loc[dim_importance["drank"] <= n_dims][["query_id", "dim"]]

    rows = np.array(selected_dims[["query_id"]].merge(q2r)["row"])
    cols = np.array(selected_dims["dim"])

    mask = np.zeros_like(qembs)
    mask[rows, cols] = 1
    enc_queries = np.where(mask, qembs, 0)

    ip, idx = index.search(enc_queries, 10) ## modificato qui
    nqueries = len(ip)

    out = []
    for i in range(nqueries):
        local_run = pd.DataFrame({"query_id": queries.iloc[i]["query_id"], "doc_id": idx[i], "score": ip[i]})
        local_run.sort_values("score", ascending=False, inplace=True)
        local_run['doc_id'] = local_run['doc_id'].apply(lambda x: mapper[x])
        out.append(local_run)

    out = pd.concat(out)

    #res = compute_measure(out, qrels, k)
    #res["alpha0"] = alpha
    return out

def retrieval_pipeline(args, index, hyperparams):

    ### ---------------- LOAD STUFF ---------------------
    # read the queries
    query_reader_params = {'sep': "\t", 'names': ["query_id", "text"], 'header': None, 'dtype': {"query_id": str}}
    queries = pd.read_csv(f"{args.datadir}/queries/{args.collection}/queries.tsv", **query_reader_params)
    # read qrels
    qrels_reader_params = {'sep': " ", 'names': ["query_id", "doc_id", "relevance"], "usecols": [0,2,3],
                            'header': None, "dtype": {"query_id": str, "doc_id": str}}
    qrels = pd.read_csv(f"{args.datadir}/qrels/{args.collection}/qrels.txt", **qrels_reader_params)
    #print('Number of og queries: ', len(queries.query_id.unique()))
    #print('Number of og queries in qrels: ', len(qrels.query_id.unique()))
    # keep only queries with relevant docs
    queries = queries.loc[queries.query_id.isin(qrels.query_id)]
    #print('Final n.queries: ', len(queries))

    # load memmap for the corpus
    corpora_memmapsdir = f"{args.datadir}/memmaps/{args.model_name}/corpora/{collection2corpus[args.collection]}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/corpus.dat",
                                        f"{corpora_memmapsdir}/corpus_mapping.csv")
    
    memmapsdir = f"{args.datadir}/memmaps/{args.model_name}/{args.collection}"
    qrys_encoder = MemmapQueriesEncoding(f"{memmapsdir}/queries.dat",
                                        f"{memmapsdir}/queries_mapping.tsv")
    
    mapping = pd.read_csv(f"{corpora_memmapsdir}/corpus_mapping.csv", dtype={'did': str})
    mapper = mapping.set_index('offset').did.to_list()

    ### ---------------- Filter selection  ---------------------
    if args.filter_function == "GPTFilter":
        #print('Load LLM DIME')
        model = SentenceTransformer(m2hf[args.model_name])
        answers_path = f"{args.datadir}/runs/{args.collection}/chatgpt4_answer.csv"
        filtering = dimension_filters.GPTFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, model=model, answers_path=answers_path)
  
    elif args.filter_function == "OracularFilter":
        #print('Load Oracle DIME')
        filtering = dimension_filters.OracularFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder)
    
    elif args.filter_function == "TopkFilter":
        #print('Load PRF DIME')
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank", "score"], 'usecols': [0, 2, 3, 4], 'sep': "\t",
                        'dtype': {"query_id": str, "doc_id": str, "rank": int, "score": float}}
        run = pd.read_csv(f"{args.datadir}/runs/{args.collection}/{args.model_name}.tsv", **run_reader_parmas)
        kpos = 0 ## 0 = single doc, 1 = average of two and so on
        filtering = dimension_filters.TopkFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, k=kpos)

    elif args.filter_function == "PRFEclipse":
        #print('Load PRF Eclipse')
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank", "score"], 'usecols': [0, 2, 3, 4], 'sep': "\t",
                        'dtype': {"query_id": str, "doc_id": str, "rank": int, "score": float}}
        run = pd.read_csv(f"{args.datadir}/runs/{args.collection}/{args.model_name}.tsv", **run_reader_parmas)
        filtering = dimension_filters.PRFEclipse(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, hyperparams=hyperparams)

    elif args.filter_function == "LLMEclipse":
        #print('Load LLM Eclipse')
        model = SentenceTransformer(m2hf[args.model_name])
        answers_path = f"{args.datadir}/runs/{args.collection}/chatgpt4_answer.csv"
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank", "score"], 'usecols': [0, 2, 3, 4], 'sep': "\t",
                        'dtype': {"query_id": str, "doc_id": str, "rank": int, "score": float}}
        run = pd.read_csv(f"{args.datadir}/runs/{args.collection}/{args.model_name}.tsv", **run_reader_parmas)
        filtering = dimension_filters.LLMEclipse(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, hyperparams=hyperparams,
                                                 model=model, answers_path=answers_path)

    else:
        #print('No filter function defined')
        pass


    rel_dims = filtering.filter_dims(queries, explode=True)
    qembs = qrys_encoder.get_encoding(queries.query_id.to_list()) 
    q2r = pd.DataFrame({"query_id": queries.query_id.to_list(), "row": np.arange(len(queries.query_id.to_list()))})

    alphas = np.round(np.arange(0.1, 1.1, 0.1), 2)
    result = []
    for alpha in alphas:
        retrieval_list = masked_retrieve_and_evaluate(queries, qrels, qembs, qrys_encoder, mapper, q2r, rel_dims, alpha, index)
        
        recall_rel1 = compute_recall(retrieval_list, qrels, label_list = [1])
        recall_rel2 = compute_recall(retrieval_list, qrels, label_list = [2])
        if args.collection != "robust04":
            recall_rel3 = compute_recall(retrieval_list, qrels, label_list = [3])
        
        if args.collection == "robust04":
            result.append({"alpha": alpha, "recall_rel1": recall_rel1, "recall_rel2": recall_rel2})
        else:
            result.append({"alpha": alpha, "recall_rel1": recall_rel1, "recall_rel2": recall_rel2, "recall_rel3": recall_rel3})

    result = pd.DataFrame(result)
    save_filename = f"/hdd4/giuder/progetti/Eclipse/data/cikm/recall_top10/{args.collection}_{args.model_name}_{args.filter_function}_{args.config_measure_name}.csv"
    result.to_csv(save_filename, index=False)


if __name__ == "__main__":
    
    tqdm.pandas()

    parser = argparse.ArgumentParser()
    #parser.add_argument("-c", "--collection", default="deeplearning19")
    parser.add_argument("-r", "--model_name", default="contriever")
    parser.add_argument("-d", "--datadir",    default="data")
    parser.add_argument("-f", "--filter_function", default="OracularFilter")

    args = parser.parse_args()

    args.filter_function = 'LLMEclipse'  ## change this
    args.hyperparams_filename = 'LLM_rq1_V0' ## change this
    
    for collection in ['deeplearning19', 'deeplearning20', 'deeplearninghd', 'robust04']:
        args.collection = collection

        print('Load FAISS index')
        faiss_path = f"{args.datadir}/vectordb/{args.model_name}/corpora/{collection2corpus[args.collection]}/"
        index_name = "index_db.faiss"
        index = faiss.read_index(faiss_path + index_name)

        print('Load hyperparams')
        filename = f"{args.datadir}/performance/configurations/{args.hyperparams_filename}.csv"
        grid_hyperparams = pd.read_csv(filename).to_dict('index')
        config = pd.read_csv('/hdd4/giuder/progetti/Eclipse/data/cikm/configurations_llmeclipse.csv',   ## change this
                            names=['collection', 'model', 'AP', 'nDCG@10'], 
                            header=0)

        for config_measure_name in ['AP', 'nDCG@10']: 
            args.config_measure_name = config_measure_name
            args.hyperparams_number =  config[(config.collection == args.collection) & (config.model == args.model_name)][config_measure_name].values[0]
            hyperparams = grid_hyperparams[args.hyperparams_number]
            retrieval_pipeline(args, index, hyperparams)


     
    
        