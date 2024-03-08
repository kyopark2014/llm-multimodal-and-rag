# LangChaind을 이용하여 Claude3 LLM으로 RAG를 활용한 Chatbot 만들기

LLM (Large Language Models)을 이용한 어플리케이션을 개발할 때에 [LangChain](https://www.langchain.com/)을 이용하면 쉽고 빠르게 개발할 수 있습니다. 하지만, 근래에 다양한 LLM 모델이 출현하고, 관련된 기술이 빠르게 발전하고 있어서, LangChain도 빠르게 진화하고 있습니다. [Anthropic Claude3](https://aws.amazon.com/ko/blogs/machine-learning/unlocking-innovation-aws-and-anthropic-push-the-boundaries-of-generative-ai-together/)는 이전 모델 대비 훨씬 빠른 속도와 높은 정확도를 가지고 있지만, Langchain의 [Bedrock](https://python.langchain.com/docs/integrations/llms/bedrock)을 더이상 사용할 수 없고, [BedrockChat](https://python.langchain.com/docs/integrations/chat/bedrock)을 사용하여야 합니다. 여기에서는 BedrockChat을 활용하여 Claude3으로 RAG를 활용하는 방법에 대해 설명합니다. 

## Architecture 개요

여기서는 서버리스 Architecture를 이용하여 RAG가 적용된 Chatbot 인프라를 구성합니다. AWS CDK로 관련된 인프라를 배포하고 편리하게 관리할 수 있습니다. 

- Multi-Region LLM: [분당 Request와 토큰 수](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html)의 제한을 완화하기 위하여 여러 Region의 LLM을 활용합니다.
- RAG 구성: OpenSearch의 Vector 검색을 이용하여 빠르고 성능이 우수한 RAG를 구성할 수 있습니다.
- 인터넷 검색: RAG에 관련된 문서가 없을 경우에 Google으 Search API를 활용하여 검색된 결과를 활용합니다.
- Prority Search: RAG의 Retrieve를 이용하여 k개의 문서를 얻었지만 일부는 관련도가 낮을수 있어 정확도에 나쁜 영향을 줄 수 있습니다. Faiss의 Similarity Search로 관련된 문서(Relevant Documents)를 관련도에 따라 정렬하고 관련이 없는 문서는 제외할 수 있습니다.
- 채팅 이력의 저장 및 활용: 서버리스 서비스인 Lambda가 실행될 때에 DynamoDB에 저장된 채팅 이력을 가져와 활용합니다.
- 지속적인 대화: API Gateway를 이용하여 WebSocket을 구성하므로써 양방향 대화를 구현할 수 있습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/6523d767-4dfb-4268-82e3-f064c91b377e)



## 주요 시스템 구성

### Claude3 API

Claude3부터는  Parameter의 경우에 max_tokens_to_sample이 max_tokens로 변경되었습니다.

```python
import boto3
from langchain_community.chat_models import BedrockChat

boto3_bedrock = boto3.client(
    service_name = 'bedrock-runtime',
    region_name = bedrock_region,
    config = Config(
        retries = {
            'max_attempts': 30
        }
    )
)

HUMAN_PROMPT = "\n\nHuman:"
parameters = {
    "max_tokens": maxOutputTokens,
    "temperature": 0.1,
    "top_k": 250,
    "top_p": 0.9,
    "stop_sequences": [HUMAN_PROMPT]
}

llm = BedrockChat(
    model_id = modelId,
    client = boto3_bedrock,
    streaming = True,
    callbacks = [StreamingStdOutCallbackHandler()],
    model_kwargs = parameters,
)
```


다수의 RAG 문서를 S3에 업로드할때 원할한 처리를 위한 Event driven architecture입니다. RAG용 문서는 채팅 UI에서 파일업로드 버튼을 통해 업로드 할 수 있지만, S3 console 또는 AWS CLI를 이용해 S3에 직접 업로드 할 수 있습니다. 이때, OpenSearch에 문서를 업로드하는 시간보다 더 빠르게 문서가 올라오는 경우에 Queue를 통해 S3 putEvent를 관리하여야 합니다. OpenSearch에 문서 업로드시에 Embedding이 필요하므로 아래와 같이 Multi-Region의 Bedrcok Embedding을 활용합니다. 

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/7403e19b-20ca-437b-b2db-725d3c57b4f3)


### RAG를 활용하기

OpenSearch로 얻어진 관련된 문서들로 부터 Context를 얻습니다. 새로운 질문(Revised question)이 한국어/영어이면 다른 Prompt를 활용하빈다. 여기서는 <context></context> tag를 활용하여 context를 다른 문장과 구분하여 더 명확하게 LLM에게 전달할 수 있습니다. 이때, readStreamMsg()을 이용하여 얻어진 stream을 client로 전달합니다. 

```python
def query_using_RAG_context(connectionId, requestId, chat, context, revised_question):    
    if isKorean(revised_question)==True:
        system = (
            """다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다.
            
            <context>
            {context}
            </context>"""
        )
    else: 
        system = (
            """Here is pieces of context, contained in <context> tags. Provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
                   
    chain = prompt | chat
    
    stream = chain.invoke(
        {
            "context": context,
            "input": revised_question,
        }
    )
    msg = readStreamMsg(connectionId, requestId, stream.content)    

    return msg

def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            sendMessage(connectionId, result)
    return msg
```


### Google Search API 활용

Google Search API를 활용하기 위해서는 [google_api_key](https://developers.google.com/custom-search/docs/paid_element?hl=ko#api_key)와 [google_cse_id](https://programmablesearchengine.google.com/controlpanel/create?hl=ko)가 필요합니다. 이 값을 코드에 하드코딩하지 않기 위하여 AWS Secret Manager를 이용합니다. 아래와 같이 google_api_key와 google_cse_id을 가져옵니다. 

```python
googleApiSecret = os.environ.get('googleApiSecret')
secretsmanager = boto3.client('secretsmanager')
try:
    get_secret_value_response = secretsmanager.get_secret_value(
        SecretId=googleApiSecret
    )
    secret = json.loads(get_secret_value_response['SecretString'])
    google_api_key = secret['google_api_key']
    google_cse_id = secret['google_cse_id']

except Exception as e:
    raise e
```

OpenSearch에 검색했을때 관련된 문서가 없거나 관련도가 낮은 경우에 아래와 같이 Google Search API로 관련된 문서를 가져와서 RAG처럼 활용합니다.

```python
if len(selected_relevant_docs)==0:  # google api
    api_key = google_api_key
    cse_id = google_cse_id 
                
    relevant_docs = []
    service = build("customsearch", "v1", developerKey=api_key)
    result = service.cse().list(q=revised_question, cx=cse_id).execute()

    if "items" in result:
        for item in result['items']:
            api_type = "google api"
            excerpt = item['snippet']
            uri = item['link']
            title = item['title']
            confidence = ""
            assessed_score = ""
                            
            doc_info = {
                "rag_type": 'search',
                "api_type": api_type,
                "confidence": confidence,
                "metadata": {
                    "source": uri,
                    "title": title,
                    "excerpt": excerpt,
                },
                "assessed_score": assessed_score,
            }
            relevant_docs.append(doc_info)           
                
    if len(relevant_docs)>=1:
        selected_relevant_docs = priority_search(revised_question, relevant_docs, bedrock_embedding, minDocSimilarity)
```

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 Seoul 리전 (ap-northeast-2)을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. [CDK 구현 코드](./cdk-llm-claude3/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다. 

## 실행결과

### RAG 활용하기

"Conversation Type"에서 [RAG - opensearch]을 선택한 후에, [error_code.pdf](./contents/error_code.pdf)을 다운로드하여 채팅창 아래의 파일 아이콘을 이용하여 업로드합니다. 이후 아래처럼 채팅창에 "보일러 에러코드에 대해 설명하여 주세요."을 입력하면, 아래와 같은 결과를 얻을 수 있습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/d962beb6-f72f-4732-a9a6-583d51fd6c3c)


### 코드 요약하기

[lambda_function.py](./lambda-chat-ws/lambda_function.py)을 다운로드 후에 채팅창 아래의 파일 아이콘을 선택하여 업로드합니다. lambda_function.py가 가지고 있는 함수들에 대한 요약을 보여줍니다.

이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/eee5b660-cfcd-46ba-a21d-b8ce312efe3c)


### 문장 오류 확인

"Conversation Type"으로 [Grammer Error Correction]을 선택하고, "다음의 문문장에서 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요."로 입력했을때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/63294a0f-f806-43be-bfed-ad9b140a0dde)

"In the following sentence, find the error in the sentence and aa explain it, and add the corrected sentence at the end of your answer."로 입력했을 때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/dfa3a8a0-2557-4a2f-9fca-6e531557725d)


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://ap-northeast-2.console.aws.amazon.com/apigateway/main/apis?region=ap-northeast-2)로 접속하여 "api-chatbot-for-llm-claude-with-rag", "api-llm-claude-with-rag"을 삭제합니다.

2) [Cloud9 Console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.


```text
cd ~/environment/rag-code-generation/cdk-llm-claude3/ && cdk destroy --all
```




## 결론

Anthropic Claude3.0은 기존 2.1보다 빠르고 가격도 경쟁력이 있습니다. 모델 성능이 대폭 개선되었고 이미지를 처리할 수 있습니다. 여기서는 Claude3.0을 LangChain을 이용하여 홀용하기 위하여 BedrockChat을 활용하였고, Chain을 이용하여 Prompt를 구성하는 방법에 대해 설명하였습니다. 또한 OpenSearch를 이용하여 RAG를 구성하고 대규모로 문서를 처리하기 위한 event driven architecture에 대해 설명하였습니다. 
