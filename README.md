# simple-RAG

[![Check Code](https://github.com/ValMobYKang/simple-RAG/actions/workflows/check_code.yml/badge.svg)](https://github.com/ValMobYKang/simple-RAG/actions/workflows/check_code.yml)

A local RAG application that applies llamaindex.

# issues:

## Reranker and Phoenix
the score from the reranker is numpy.float32. Phoenix doesn't support this type. So if we change the node.score = score to `node.score = score.item()`, t will be fixed: https://github.com/run-llama/llama_index/blob/adbf60f22601866467964152262d07329dc95ee0/llama_index/indices/postprocessor/sbert_rerank.py#L70C30-L70C35