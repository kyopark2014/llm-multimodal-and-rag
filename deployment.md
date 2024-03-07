# 인프라 설치하기

## Bedrock 사용 권한 설정하기

Bedrock은 Virginia(us-east-1)과 Oregon(us-west-2) 리전을 사용합니다. [Model access - Virginia](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess)과 [Model access - Oregon](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)에 접속해서 [Edit]를 선택하여 모든 모델을 사용할 수 있도록 설정합니다. 특히 Anthropic Claude와 "Titan Embeddings G1 - Text"은 LLM 및 Vector Embedding을 위해서 반드시 사용이 가능하여야 합니다.



## CDK를 이용한 인프라 설치하기


여기서는 [Cloud9](https://aws.amazon.com/ko/cloud9/)에서 [AWS CDK](https://aws.amazon.com/ko/cdk/)를 이용하여 인프라를 설치합니다.

1) [Cloud9 Console](https://ap-northeast-1.console.aws.amazon.com/cloud9control/home?region=ap-northeast-1#/create)에 접속하여 [Create environment]-[Name]에서 “chatbot”으로 이름을 입력하고, EC2 instance는 “m5.large”를 선택합니다. 나머지는 기본값을 유지하고, 하단으로 스크롤하여 [Create]를 선택합니다.

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/7c20d80c-52fc-4d18-b673-bd85e2660850)

2) [Environment](https://ap-northeast-1.console.aws.amazon.com/cloud9control/home?region=ap-northeast-1#/)에서 “chatbot”를 [Open]한 후에 아래와 같이 터미널을 실행합니다.

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/b7d0c3c0-3e94-4126-b28d-d269d2635239)

3) EBS 크기 변경

아래와 같이 스크립트를 다운로드 합니다. 

```text
curl https://raw.githubusercontent.com/kyopark2014/technical-summary/main/resize.sh -o resize.sh
```

이후 아래 명령어로 용량을 80G로 변경합니다.
```text
chmod a+rx resize.sh && ./resize.sh 80
```


4) 소스를 다운로드합니다.

```java
git clone https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock
```

5) cdk 폴더로 이동하여 필요한 라이브러리를 설치합니다.

```java
cd korean-chatbot-using-amazon-bedrock/cdk-korean-chatbot/ && npm install
```

6) CDK 사용을 위해 Boostraping을 수행합니다.

아래 명령어로 Account ID를 확인합니다.

```java
aws sts get-caller-identity --query Account --output text
```

아래와 같이 bootstrap을 수행합니다. 여기서 "account-id"는 상기 명령어로 확인한 12자리의 Account ID입니다. bootstrap 1회만 수행하면 되므로, 기존에 cdk를 사용하고 있었다면 bootstrap은 건너뛰어도 됩니다.

```java
cdk bootstrap aws://account-id/ap-northeast-1
```

7) 인프라를 설치합니다.

```java
cdk deploy --all
```

설치가 완료되면 아래와 같은 Output이 나옵니다. 

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/766b4d33-13c9-49b1-9462-832120f73109)

8) HTMl 파일을 S3에 복사합니다.

아래와 같이 Output의 HtmlUpdateCommend을 붙여넣기 합니다. 

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/130bae3d-3bde-43a8-8bcb-55cee1e06e42)

9) Google API Key Update하기

[api_key](https://developers.google.com/custom-search/docs/paid_element?hl=ko#api_key)에서 [키 가져오기] - [Select or create project]를 선택하여 Google API Key를 가져옵니다. 만약 기존 키가 없다면 새로 생성합니다.

[새 검색엔진 만들기](https://programmablesearchengine.google.com/controlpanel/create?hl=ko)에서 검색엔진을 설정합니다. 이때, 검색할 내용은 "전체 웹 검색"을 선택하여야 합니다.

[Secret Console](https://us-west-2.console.aws.amazon.com/secretsmanager/secret?name=googl_api_key&region=us-west-2)에 접속하여 [Retrieve secret value]를 선택하여, google_api_key와 google_cse_id를 업데이트합니다.

10) WebUrl를 선택하여 브라우저로 접속합니다.

## [참고] S3 Bucket 이름 지정하기

Build 정보를 가지고 있는 [cdk-korean-chatbot-stack.ts](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/cdk-korean-chatbot/lib/cdk-korean-chatbot-stack.ts)에서 Bucket 이름을 지정하는 방법은 아래와 같습니다. S3의 Bucket 이름은 AWS 전세계 리전에서 유일하므로 동일한 이름을 사용하면 빌드 에러가 발생합니다. 아래와 같이 bucketName을 원하는 이름으로 변경하고 bucketName 설정하는 부분에 대한 주석을 제거합니다. 이를 변경하지 않을 경우에 cdk로 시작하는 임의의 bucket명을 자동으로 생성하여 사용하게 됩니다. 

```typescript
const bucketName = `storage-for-${projectName}-${region}`;   <-------- 1) 이름 변경

// s3 
const s3Bucket = new s3.Bucket(this, `storage-${projectName}`, {
    // bucketName: bucketName,              <----------- 2) 주석 제거
    blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    autoDeleteObjects: true,
    publicReadAccess: false,
    versioned: false,
    cors: [
        {
            allowedHeaders: ['*'],
            allowedMethods: [
                s3.HttpMethods.POST,
                s3.HttpMethods.PUT,
            ],
            allowedOrigins: ['*'],
        },
    ],
});
```

