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

import ir_measures
def compute_measure(run, qrels, measure_name):
    measure = [ir_measures.parse_measure(measure_name)]
    out = pd.DataFrame(ir_measures.iter_calc(measure, qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out

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

def masked_retrieve_and_evaluate(queries, qrels, qembs, qrys_encoder, mapper, q2r, dim_importance, alpha, measure, index):
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

    ip, idx = index.search(enc_queries, 1000)
    nqueries = len(ip)

    out = []
    for i in range(nqueries):
        local_run = pd.DataFrame({"query_id": queries.iloc[i]["query_id"], "doc_id": idx[i], "score": ip[i]})
        local_run.sort_values("score", ascending=False, inplace=True)
        local_run['doc_id'] = local_run['doc_id'].apply(lambda x: mapper[x])
        out.append(local_run)

    out = pd.concat(out)

    res = compute_measure(out, qrels, measure)
    res["alpha0"] = alpha
    return res

def retrieval_pipeline(args, index, ncomb, hyperparams):

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
        kpos = ncomb ## 0 = single doc, 1 = average of two and so on
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
    for measure_name in ['nDCG@10', 'AP']:
        #print('Working with measure: ', measure_name)
        measure_folder = 'ndcg' if measure_name == 'nDCG@10' else 'ap'

        result = []
        for alpha in alphas:
            output = masked_retrieve_and_evaluate(queries, qrels, qembs, qrys_encoder, mapper, q2r, rel_dims, alpha, measure_name, index)
            result.append(output)
            print(f'{measure_name}@{alpha}: ', output.value.mean())

        result = pd.concat(result)
        if args.load_hyperparams == 'Yes':
            for key, value in hyperparams.items():
                result[key] = value

        ftf = FilterToFolder[args.filter_function]
        save_filename = f"{args.datadir}/performance/{ftf}/{measure_folder}/{args.collection}_{args.model_name}_{args.filter_function}_{measure_name}_{ncomb}.csv"
        result.to_csv(save_filename, index=False)


def nsplit(filename):
    collection, retrieval_model, _, _, _ = filename.rsplit('/')[-1].split('_')
    return [collection, retrieval_model]

def cond(substring, long_string):
    found = True if substring in nsplit(long_string) else False
    return found

if __name__ == "__main__":
    
    tqdm.pandas()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="deeplearning19")
    parser.add_argument("-r", "--model_name", default="contriever")
    parser.add_argument("-d", "--datadir",    default="data")
    parser.add_argument("-f", "--filter_function", default="OracularFilter")
    parser.add_argument("--load_hyperparams", default="Yes")
    parser.add_argument("--hyperparams_filename", default="rq1")
    parser.add_argument("--checkpoint", default="Yes")
    parser.add_argument("--starting_index", default=0, type=int)
    parser.add_argument("--kpos", default=-1, type=int)

    args = parser.parse_args()

    print('DIME used: ', args.filter_function)

    print('Load FAISS index')
    faiss_path = f"{args.datadir}/vectordb/{args.model_name}/corpora/{collection2corpus[args.collection]}/"
    index_name = "index_db.faiss"
    index = faiss.read_index(faiss_path + index_name)

    if args.checkpoint == 'Yes':
        print('load checkpoint')
        ftf = FilterToFolder[args.filter_function]
        path_to_checkpoint = f"{args.datadir}/performance/{ftf}"
        print(path_to_checkpoint)
        ## check in /ap and /ndcg then select the correct collection and model_name files
        files = glob(f"{path_to_checkpoint}/ap/*.csv") + glob(f"{path_to_checkpoint}/ndcg/*.csv")
        files = [file for file in files if (cond(args.collection, file)) and (cond(args.model_name, file))]
        counter = Counter([x[:-4].rsplit('_')[-1] for x in files])
        ## if for the configuration the values are less than 2 then I need to start from this config 
        ### two == ap + ndcg file
        lesser_ncomb  = min([int(number) for number, count in counter.items() if count < 2], default=None)
        if not lesser_ncomb: 
            ## is None, the last configuration have both ap and ndcg 
            print('Last configuration is full, so restart from there: ')
            lesser_ncomb = int(max(counter, key=lambda k: int(k)))

    if args.load_hyperparams == 'Yes':
        print('Load hyperparams')
        filename = f"{args.datadir}/performance/configurations/{args.hyperparams_filename}.csv"
        grid_hyperparams = pd.read_csv(filename).to_dict('index')
        if args.checkpoint == 'Yes':
            print(f'Restart at {lesser_ncomb}/{len(grid_hyperparams)}')
        else: 
            print(f"Number of combinations: {len(grid_hyperparams)}")
        for ncomb, hyperparams in tqdm(grid_hyperparams.items()): 
            if args.checkpoint == 'Yes' and int(ncomb) >= lesser_ncomb: 
                retrieval_pipeline(args, index, ncomb, hyperparams)
            elif args.checkpoint == 'No':
                #print('n. ', ncomb, ' with params: ', hyperparams)
                retrieval_pipeline(args, index, ncomb + args.starting_index, hyperparams)
            else:
                pass
    else: 
        print('No hyperparams loaded')
        if args.kpos >= 0:
            retrieval_pipeline(args, index, args.kpos, None)
        else: 
            retrieval_pipeline(args, index, 0, None)

     
    
        