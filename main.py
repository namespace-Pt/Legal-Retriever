import gc
import json
import torch

from loguru import logger
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from elasticsearch import Elasticsearch
from elasticsearch import helpers

from utils.modeling import EmbeddingModel


# embedding_model (defaults to bge-m3)
logger.info("Loading embedding model...")
with open(f"config/bge-m3.json", encoding="utf-8") as f:
    model_config = json.load(f)
model = EmbeddingModel(**model_config)
logger.info("Connecting to elasticsearch...")
elasticsearch = Elasticsearch("http://127.0.0.1:9200")

logger.info("Starting API...")
app = FastAPI()


class IndexRequest(BaseModel):
    index_name: str = "tmp"
    texts: List[str] = []


class SearchRequest(BaseModel):
    index_name: str = "tmp"
    query: str = None
    topk: int = 10


@app.get("/set-model/{model_name}")
def set_model(model_name: str):
    global model
    if model is not None:
        del model
        torch.cuda.empty_cache()
        gc.collect()
    with open(f"config/{model_name}.json", encoding="utf-8") as f:
        model_config = json.load(f)
    model = EmbeddingModel(**model_config)
    output = {"model_config": model_config}
    return JSONResponse(output)


@app.post("/index/")
async def index(req: IndexRequest):
    texts = req.texts
    index_name = req.index_name
    
    ###### 1. create index if it does not exist
    existence = elasticsearch.indices.exists(index=index_name).body
    if not existence:
        elasticsearch.indices.create(
            index=index_name,
            settings={
                # 'analysis': {
                #     'analyzer': {
                #         # we must set the default analyzer
                #         "default": {
                #             "type": "smartcn"
                #         }
                #     }
                # },
                # "index.mapping.ignore_malformed": True
            },
            mappings={
                "properties": {
                    # field name
                    "country": {
                        "type": "keyword",
                    },
                    "code": {
                        "type": "keyword",
                    },
                    "content": {
                        "type": "text",
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": model.ndim,
                        # enable hnsw
                        "index": True,
                        # inner product only allows unit-length vector
                        "similarity": "dot_product"  
                    }
                }
            }
        )
    
    ##### 2. index documents
    def generate_doc():
        for i, text in enumerate(texts):
            embedding = embeddings[i]
            yield {
                "_index": index_name,
                "content": text,
                "embedding": embedding.tolist()
            }
    embeddings = model.encode(texts, field="key", do_template=True)
    num_docs, _ = helpers.bulk(elasticsearch, generate_doc())
    
    assert num_docs == len(texts)
    output = {"num_docs": num_docs}
    return JSONResponse(output)


@app.post("/search/")
async def search(req: SearchRequest):
    query = req.query
    index_name = req.index_name
    topk = req.topk
    
    resp = elasticsearch.search(
        index=index_name, 
        # ignore embedding in the returned source
        _source={
            "excludes": "embedding",
        },
        query={
            "match": 
                {
                    "content": {
                        "query": query,
                        "boost": 0.1,
                    }
                }
        },
        knn={
            "field": "embedding",
            "query_vector": model.encode(query).tolist(),  # generate embedding for query so it can be compared to `title_vector`
            "k": topk,
            "num_candidates": 100,
            "boost": 0.9
        },
        size=topk,
    )
    
    hits = resp["hits"]["hits"]

    output = {
        "docs": [hit["_source"] for hit in hits],
        "scores": [hit["_score"] for hit in hits],
    }
    return JSONResponse(output)


@app.get("/remove-index/{index_name}")
def remove_index(index_name: str):
    elasticsearch.indices.delete(index=index_name)
    output = {"removed_index_name": index_name}
    return JSONResponse(output)

