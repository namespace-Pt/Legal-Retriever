import os
import json
import torch
import subprocess
import datasets
from transformers import AutoTokenizer
from typing import List, Optional, Union
from tqdm import tqdm
from .utils import rmdirs


class BM25Retriever:
    def __init__(self, anserini_dir: str, k1: float = 0.9, b: float = 0.4, tokenizer_kwargs: dict = {"pretrained_model_name_or_path": "Qwen/Qwen2-0.5B"}) -> None:
        self.anserini_dir = anserini_dir
        self.k1 = k1
        self.b = b
        # Use qwen tokenizer as a placeholder because we need the tokenizer when applying the template
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)

    def _prepare_collection(self, corpus: datasets.Dataset, collection_dir: str, max_docs_per_file: int = 1000000):
        """Convert corpus into Anserini readable collections."""
        rmdirs(collection_dir)
        file_index = 0

        # use DataLoader to accelerate reading corpus file (using for loop is slow because of the template)
        dataloader = torch.utils.data.DataLoader(
            corpus,
            batch_size=512,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=32,
        )

        i = 0
        for x in tqdm(dataloader, desc="Preparing Anserini Collection"):
            texts = x["text"]
            for text in texts:
                if i % max_docs_per_file == 0:
                    if i > 0:
                        output_jsonl_file.close()
                    output_path = os.path.join(collection_dir, 'docs{:02d}.json'.format(file_index))
                    output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                    file_index += 1
                output_dict = {'id': i, 'contents': text}
                output_jsonl_file.write(json.dumps(output_dict) + '\n')
                i += 1
        output_jsonl_file.close()
        assert i == len(corpus)

    def _prepare_query(
        self, 
        inputs: Union[str, List[str]], 
        query_dir: str, 
        max_queries_per_file: int=10000
    ):
        """Convert raw query strings into Anserini readable files."""
        rmdirs(query_dir)

        if isinstance(inputs, str):
            queries = [inputs]
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            queries = inputs
        else:
            raise ValueError(f"Inputs of type {type(inputs)} not supported!")

        for i, query in enumerate(queries):
            # NOTE: repr query because it may contain newline character
            query = repr(query)[1:-1]
            if len(query.strip()) == 0:
                query = "EMPTY_QUERY"
            queries[i] = query

        # we must split large query file into smaller segments for efficiency
        query_paths = []
        if len(queries) > max_queries_per_file:
            # split queries into shards because Anserini cannot deal with large query file
            for idx, query in enumerate(queries):
                if idx % max_queries_per_file == 0:
                    if idx > 0:
                        g.close()
                    query_path = os.path.join(query_dir, f"queries.{str(idx // max_queries_per_file)}.tsv")
                    g = open(query_path, "w")
                    query_paths.append(query_path)
                g.write("\t".join([str(idx), query]) + "\n")
            g.close()
        else:
            query_path = os.path.join(query_dir, "queries.tsv")
            with open(query_path, "w") as f:
                for idx, query in enumerate(queries):
                    f.write("\t".join([str(idx), query]) + "\n")
            query_paths.append(query_path)
        return query_paths
    
    def _prepare_result(self, query_path, result_path):
        retrieval_results = {}
        with open(query_path) as f, open(result_path) as g:
            for i, line in enumerate(f):
                qid = int(line.strip().split("\t")[0])
                if i == 0:
                    first = qid
                last = qid

            for i, line in enumerate(tqdm(g, desc="Collecting Retrieval Results")):
                fields = line.strip().split("\t")
                qid = int(fields[0])
                tidx = int(fields[1])
                if qid not in retrieval_results:
                    retrieval_results[qid] = []
                retrieval_results[qid].append(tidx)

        # NOTE: the returned retrieval_indices must follow the order of the input queries
        # NOTE: Anserini may silently skip some queries, we append -1 to all these queries 
        retrieval_indices = []
        # NOTE: last + 1 to enumerate all ids
        for i in range(first, last + 1):
            if i in retrieval_results:
                retrieval_indices.append(retrieval_results[i])
            else:
                retrieval_indices.append([-1])
        # TODO: get retrieval_scores
        return retrieval_indices, None
    
    def index(
        self, 
        corpus: Union[datasets.Dataset, List[str]], 
        threads: int = 32, 
        language: str = "en", 
        storeDocvectors: bool = False, 
        load_collection: bool = False, 
        load_index: bool = False,
        output_dir: str = "data/results", 
        max_docs_per_file: int = 1000000,
        **kwargs,
    ):
        index_dir = os.path.join(output_dir, "index")
        collection_dir = os.path.join(output_dir, "collection")
        self.output_dir = output_dir
        self.language = language

        if not load_collection and not load_index:
            self._prepare_collection(corpus, collection_dir, max_docs_per_file=max_docs_per_file)           

        if not load_index:
            rmdirs(index_dir)
            args = [
                f"sh {self.anserini_dir}/target/appassembler/bin/IndexCollection -collection JsonCollection -generator DefaultLuceneDocumentGenerator",
                f"-input {collection_dir} -index {index_dir} -threads {threads} -language {language}",
                "-storeDocvectors" if storeDocvectors else ""
            ]
            subprocess.run(" ".join(args), shell=True)
        
    def search(
        self, 
        inputs: Union[str, List[str], datasets.Dataset], 
        hits: int = 100, 
        k1: Optional[float] = None, 
        b: Optional[float] = None, 
        language: Optional[str] = None, 
        threads: int = 32, 
        parallelism: int = 4, 
        output_dir: Optional[str] = None, 
        batch_size: int = 10000,
        show_progress: Optional[bool] = None,
        **kwargs,
    ):
        if k1 is None:
            k1 = self.k1
        if b is None:
            b = self.b
        if output_dir is None and not hasattr(self, "output_dir"):
            raise ValueError(f"Make sure there is an index by either calling .index() or specifying an existing index with index_dir=xxx!")
        elif output_dir is None:
            output_dir = self.output_dir
        if language is None:
            language = self.language

        index_dir = os.path.join(output_dir, "index")
        query_dir = os.path.join(output_dir, "query")

        dataloader = torch.utils.data.DataLoader(
            inputs,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=0,
        )

        retrieval_indices = []
        # TODO: get retrieval_scores
        retrieval_scores = None

        for batch_inputs in tqdm(dataloader, desc="Searching", disable=not show_progress):
            # NOTE: in case the inputs is a dataset, we slice out the query column
            if isinstance(batch_inputs, dict):
                batch_inputs = batch_inputs["query"]

            query_paths = self._prepare_query(batch_inputs, query_dir, max_queries_per_file=batch_size)

            for query_path in tqdm(query_paths, desc="Searching"):
                tmp_result_path = query_path + ".tmp"
                args = [
                    f"sh {self.anserini_dir}/target/appassembler/bin/SearchCollection -topicreader TsvString -format msmarco",
                    f"-index {index_dir} -topics {query_path} -output {tmp_result_path} -bm25 -bm25.k1 {k1} -bm25.b {b}",
                    f"-hits {hits} -threads {threads} -parallelism {parallelism} -language {language}"
                ]
                subprocess.run(" ".join(args), shell=True)
                indices, _ = self._prepare_result(query_path, tmp_result_path)
                retrieval_indices.extend(indices)

        return retrieval_indices, retrieval_scores

    def rerank(self, *args, **kwargs):
        raise NotImplementedError
