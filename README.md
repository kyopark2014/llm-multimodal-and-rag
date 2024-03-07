# Claud3을 이용한 Chatbot 만들기


여기에서는 Clade3를 이용한 Chatbot을 만드립니다. Cluade2.1, Claude Instant 모델과 비교도 할 수 있습니다.

![image](https://github.com/kyopark2014/llm-chatbot-using-claude3/assets/52392004/6523d767-4dfb-4268-82e3-f064c91b377e)

## Claude3 API

Claude3부터는 Langchain의 [Bedrock](https://python.langchain.com/docs/integrations/llms/bedrock)을 더이상 사용할 수 없고, [BedrockChat](https://python.langchain.com/docs/integrations/chat/bedrock)을 사용하여야 합니다. Parameter의 경우에 max_tokens_to_sample이 max_tokens로 변경되었습니다.

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


