# Hybrid Search를 위한 Ensemble

Multi-RAG의 경우에 priority search를 통해 Vector/Keyword 검색을 Vector 기준으로 similiarity를 평가하였으나, OpenSearch만을 사용하는 경우에 Ensemble Retriever를 이용할 수 있습니다.

## LangChain의 Ensemble

[LangChain의 Ensemble Retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/ensemble/)은 아래와 같이 RRF방식의 retriever들을 weight를 이용해 적용할 수 있습니다. 

```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
```

## OpenSearch의 Lexical Retriever

[Retriever for lexical search](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/10_advanced_question_answering/03_2_rag_opensearch_hybrid_ensemble_retriever_kr.ipynb)을 참조하여 Lexical 검색을 위한 Retriever를 정의합니다. 

[OpenSearchLexicalSearchRetriever](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/utils/rag.py)는 아래와 같습니다.

```python
class OpenSearchLexicalSearchRetriever(BaseRetriever):

    os_client: Any
    index_name: str
    k = 3
    minimum_should_match = 0
    filter = []

    def normalize_search_results(self, search_results):

        hits = (search_results["hits"]["hits"])
        max_score = float(search_results["hits"]["max_score"])
        for hit in hits:
            hit["_score"] = float(hit["_score"]) / max_score
        search_results["hits"]["max_score"] = hits[0]["_score"]
        search_results["hits"]["hits"] = hits
        return search_results

    def update_search_params(self, **kwargs):

        self.k = kwargs.get("k", 3)
        self.minimum_should_match = kwargs.get("minimum_should_match", 0)
        self.filter = kwargs.get("filter", [])
        self.index_name = kwargs.get("index_name", self.index_name)

    def _reset_search_params(self, ):

        self.k = 3
        self.minimum_should_match = 0
        self.filter = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        query = opensearch_utils.get_query(
            query=query,
            minimum_should_match=self.minimum_should_match,
            filter=self.filter
        )
        query["size"] = self.k

        print ("lexical search query: ")
        pprint(query)

        search_results = opensearch_utils.search_document(
            os_client=self.os_client,
            query=query,
            index_name=self.index_name
        )

        results = []
        if search_results["hits"]["hits"]:
            search_results = self.normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:

                metadata = res["_source"]["metadata"]
                metadata["id"] = res["_id"]

                doc = Document(
                    page_content=res["_source"]["text"],
                    metadata=metadata
                )
                results.append((doc))

        self._reset_search_params()

        return results[:self.k]
```




