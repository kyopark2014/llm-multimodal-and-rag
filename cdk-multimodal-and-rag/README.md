# CDK로 배포하기

S3를 생성합니다.

```typescript
const s3Bucket = new s3.Bucket(this, `storage-${projectName}`, {
    bucketName: bucketName,
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

빌드후 Output에 복사 명령어를 편의상 출력합니다.

```typescript
new cdk.CfnOutput(this, 'HtmlUpdateCommend', {
    value: 'aws s3 cp ../html/ ' + 's3://' + s3Bucket.bucketName + '/ --recursive',
    description: 'copy commend for web pages',
});
```

CloudFront를 설정합니다.

```typescript
const distribution = new cloudFront.Distribution(this, `cloudfront-for-${projectName}`, {
    defaultBehavior: {
        origin: new origins.S3Origin(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    },
    priceClass: cloudFront.PriceClass.PRICE_CLASS_200,
});
```

채팅이력을 저장하기 위하여 DynamoDB를 준비합니다.

```typescript
const callLogTableName = `db-call-log-for-${projectName}`;
const callLogDataTable = new dynamodb.Table(this, `db-call-log-for-${projectName}`, {
    tableName: callLogTableName,
    partitionKey: { name: 'user_id', type: dynamodb.AttributeType.STRING },
    sortKey: { name: 'request_time', type: dynamodb.AttributeType.STRING },
    billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
});
const callLogIndexName = `index-type-for-${projectName}`;
callLogDataTable.addGlobalSecondaryIndex({ // GSI
    indexName: callLogIndexName,
    partitionKey: { name: 'request_id', type: dynamodb.AttributeType.STRING },
});
```

Websocket을 위한 Lambda의 권한을 설정합니다.

```typescript
// Lambda - chat (websocket)
const roleLambdaWebsocket = new iam.Role(this, `role-lambda-chat-ws-for-${projectName}`, {
    roleName: `role-lambda-chat-ws-for-${projectName}-${region}`,
    assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com")
    )
});
roleLambdaWebsocket.addManagedPolicy({
    managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
});
const BedrockPolicy = new iam.PolicyStatement({  // policy statement for sagemaker
    resources: ['*'],
    actions: ['bedrock:*'],
});
roleLambdaWebsocket.attachInlinePolicy( // add bedrock policy
    new iam.Policy(this, `bedrock-policy-lambda-chat-ws-for-${projectName}`, {
        statements: [BedrockPolicy],
    }),
);
const apiInvokePolicy = new iam.PolicyStatement({
    resources: ['*'],
    actions: [
        'execute-api:Invoke',
        'execute-api:ManageConnections'
    ],
});
roleLambdaWebsocket.attachInlinePolicy(
    new iam.Policy(this, `api-invoke-policy-for-${projectName}`, {
        statements: [apiInvokePolicy],
    }),
);  
```

OpenSearch를 설치합니다.

```typescript
// opensearch
// Permission for OpenSearch
const domainName = projectName
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const resourceArn = `arn:aws:es:${region}:${accountId}:domain/${domainName}/*`
if (debug) {
    new cdk.CfnOutput(this, `resource-arn-for-${projectName}`, {
        value: resourceArn,
        description: 'The arn of resource',
    });
}

const OpenSearchAccessPolicy = new iam.PolicyStatement({
    resources: [resourceArn],
    actions: ['es:*'],
    effect: iam.Effect.ALLOW,
    principals: [new iam.AnyPrincipal()],
});

const domain = new opensearch.Domain(this, 'Domain', {
    version: opensearch.EngineVersion.OPENSEARCH_2_3,

    domainName: domainName,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    enforceHttps: true,
    fineGrainedAccessControl: {
        masterUserName: opensearch_account,
        // masterUserPassword: cdk.SecretValue.secretsManager('opensearch-private-key'),
        masterUserPassword: cdk.SecretValue.unsafePlainText(opensearch_passwd)
    },
    capacity: {
        masterNodes: 3,
        masterNodeInstanceType: 'r6g.large.search',
        // multiAzWithStandbyEnabled: false,
        dataNodes: 15,
        dataNodeInstanceType: 'r6g.large.search',
        // warmNodes: 2,
        // warmInstanceType: 'ultrawarm1.medium.search',
    },
    accessPolicies: [OpenSearchAccessPolicy],
    ebs: {
        volumeSize: 100,
        volumeType: ec2.EbsDeviceVolumeType.GP3,
    },
    nodeToNodeEncryption: true,
    encryptionAtRest: {
        enabled: true,
    },
    zoneAwareness: {
        enabled: true,
        availabilityZoneCount: 3,
    }
});
new cdk.CfnOutput(this, `Domain-of-OpenSearch-for-${projectName}`, {
    value: domain.domainArn,
    description: 'The arm of OpenSearch Domain',
});
new cdk.CfnOutput(this, `Endpoint-of-OpenSearch-for-${projectName}`, {
    value: 'https://' + domain.domainEndpoint,
    description: 'The endpoint of OpenSearch Domain',
});
opensearch_url = 'https://' + domain.domainEndpoint;
```

API를 Gateway를 준비합니다. 

```typescript
// api role
const role = new iam.Role(this, `api-role-for-${projectName}`, {
    roleName: `api-role-for-${projectName}-${region}`,
    assumedBy: new iam.ServicePrincipal("apigateway.amazonaws.com")
});
role.addToPolicy(new iam.PolicyStatement({
    resources: ['*'],
    actions: [
        'lambda:InvokeFunction',
        'cloudwatch:*'
    ]
}));
role.addManagedPolicy({
    managedPolicyArn: 'arn:aws:iam::aws:policy/AWSLambdaExecute',
});

// API Gateway
const api = new apiGateway.RestApi(this, `api-chatbot-for-${projectName}`, {
    description: 'API Gateway for chatbot',
    endpointTypes: [apiGateway.EndpointType.REGIONAL],
    binaryMediaTypes: ['application/pdf', 'text/plain', 'text/csv', 'application/vnd.ms-powerpoint', 'application/vnd.ms-excel', 'application/msword'],
    deployOptions: {
        stageName: stage,

        // logging for debug
        // loggingLevel: apiGateway.MethodLoggingLevel.INFO, 
        // dataTraceEnabled: true,
    },
});  
```

파일업로드시 presigned url을 가져오기 위한 Lambda를 생성합니다.

```typescript
// Lambda - Upload
const lambdaUpload = new lambda.Function(this, `lambda-upload-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-upload-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-upload"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(10),
    environment: {
        bucketName: s3Bucket.bucketName,
        s3_prefix: s3_prefix
    }
});
s3Bucket.grantReadWrite(lambdaUpload);

// POST method - upload
const resourceName = "upload";
const upload = api.root.addResource(resourceName);
upload.addMethod('POST', new apiGateway.LambdaIntegration(lambdaUpload, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting  
distribution.addBehavior("/upload", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});    
```

Client에서 DynamoDB를 조회하기 위한 lambda를 설정합니다.

```typescript
// Lambda - queryResult
const lambdaQueryResult = new lambda.Function(this, `lambda-query-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-query-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-query"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName,
        indexName: callLogIndexName
    }
});
callLogDataTable.grantReadWriteData(lambdaQueryResult); // permission for dynamo

// POST method - query
const query = api.root.addResource("query");
query.addMethod('POST', new apiGateway.LambdaIntegration(lambdaQueryResult, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/query", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

Client에서 채팅이력을 가져오기 위한 Lambda를 설정합니다.

```typescript
// Lambda - getHistory
const lambdaGetHistory = new lambda.Function(this, `lambda-gethistory-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-gethistory-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-gethistory"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName
    }
});
callLogDataTable.grantReadWriteData(lambdaGetHistory); // permission for dynamo

// POST method - history
const history = api.root.addResource("history");
history.addMethod('POST', new apiGateway.LambdaIntegration(lambdaGetHistory, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/history", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

Client에서 채팅이력을 삭제할때에 서버의 데이터도 같이 삭제하기 위한 Lambda를 설정합니다. 

```typescript
// Lambda - deleteItems
const lambdaDeleteItems = new lambda.Function(this, `lambda-deleteItems-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-deleteItems-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-delete-items"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName
    }
});
callLogDataTable.grantReadWriteData(lambdaDeleteItems); // permission for dynamo

// POST method - delete items
const deleteItem = api.root.addResource("delete");
deleteItem.addMethod('POST', new apiGateway.LambdaIntegration(lambdaDeleteItems, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/delete", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

Websocket 방식의 API Gateway를 설정합니다. 

```typescript
// stream api gateway
// API Gateway
const websocketapi = new apigatewayv2.CfnApi(this, `ws-api-for-${projectName}`, {
    description: 'API Gateway for chatbot using websocket',
    apiKeySelectionExpression: "$request.header.x-api-key",
    name: 'api-' + projectName,
    protocolType: "WEBSOCKET", // WEBSOCKET or HTTP
    routeSelectionExpression: "$request.body.action",
});
websocketapi.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY); // DESTROY, RETAIN

const wss_url = `wss://${websocketapi.attrApiId}.execute-api.${region}.amazonaws.com/${stage}`;
new cdk.CfnOutput(this, 'web-socket-url', {
    value: wss_url,
    description: 'The URL of Web Socket',
});

const connection_url = `https://${websocketapi.attrApiId}.execute-api.${region}.amazonaws.com/${stage}`;
if (debug) {
    new cdk.CfnOutput(this, 'api-identifier', {
        value: websocketapi.attrApiId,
        description: 'The API identifier.',
    });

    new cdk.CfnOutput(this, 'connection-url', {
        value: connection_url,
        description: 'The URL of connection',
    });
}
```

Google Search API에 대한 Secret를 위한 Secret manager를 준비합니다. 

```typescript
const googleApiSecret = new secretsmanager.Secret(this, `google-api-secret-for-${projectName}`, {
    description: 'secret for google api key',
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    secretName: 'googl_api_key',
    generateSecretString: {
        secretStringTemplate: JSON.stringify({
            google_cse_id: 'cse_id'
        }),
        generateStringKey: 'google_api_key',
        excludeCharacters: '/@"',
    },

});
googleApiSecret.grantRead(roleLambdaWebsocket) 
```

WebSocket 방식으로 chat을 관리하기 위한 Lambda를 설정합니다. 

```typescript
// lambda-chat using websocket    
const lambdaChatWebsocket = new lambda.DockerImageFunction(this, `lambda-chat-ws-for-${projectName}`, {
    description: 'lambda for chat using websocket',
    functionName: `lambda-chat-ws-for-${projectName}`,
    code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-chat-ws')),
    timeout: cdk.Duration.seconds(300),
    memorySize: 8192,
    role: roleLambdaWebsocket,
    environment: {
        // bedrock_region: bedrock_region,
        // model_id: model_id,
        s3_bucket: s3Bucket.bucketName,
        s3_prefix: s3_prefix,
        callLogTableName: callLogTableName,
        connection_url: connection_url,
        enableReference: enableReference,
        opensearch_account: opensearch_account,
        opensearch_passwd: opensearch_passwd,
        opensearch_url: opensearch_url,
        path: 'https://' + distribution.domainName + '/',
        roleArn: roleLambdaWebsocket.roleArn,
        debugMessageMode: debugMessageMode,
        useParallelRAG: useParallelRAG,
        numberOfRelevantDocs: numberOfRelevantDocs,
        profile_of_LLMs: JSON.stringify(profile_of_LLMs),
        claude3_sonnet: JSON.stringify(claude3_sonnet),
        claude2: JSON.stringify(claude2),
        claude_instant: JSON.stringify(claude_instant),
        googleApiSecret: googleApiSecret.secretName,
    }
});
lambdaChatWebsocket.grantInvoke(new iam.ServicePrincipal('apigateway.amazonaws.com'));
s3Bucket.grantReadWrite(lambdaChatWebsocket); // permission for s3
callLogDataTable.grantReadWriteData(lambdaChatWebsocket); // permission for dynamo 
```

Websockt을 위한 API Gateway에 대한 설정을 수행합니다. 

```typescript
const integrationUri = `arn:aws:apigateway:${region}:lambda:path/2015-03-31/functions/${lambdaChatWebsocket.functionArn}/invocations`;
const cfnIntegration = new apigatewayv2.CfnIntegration(this, `api-integration-for-${projectName}`, {
    apiId: websocketapi.attrApiId,
    integrationType: 'AWS_PROXY',
    credentialsArn: role.roleArn,
    connectionType: 'INTERNET',
    description: 'Integration for connect',
    integrationUri: integrationUri,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-connect`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$connect",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'connect',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-disconnect`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$disconnect",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'disconnect',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-default`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$default",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'default',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnStage(this, `api-stage-for-${projectName}`, {
    apiId: websocketapi.attrApiId,
    stageName: stage
}); 
```

대규모로 문서가 업로드 될때에 S3의 이벤트를 처리하기 위한 Lambda를 설정합니다. 

```typescript
// S3 - Lambda(S3 event) - SQS(fifo) - Lambda(document)
// SQS for S3 event (fifo) 
let queueUrl: string[] = [];
let queue: any[] = [];
for (let i = 0; i < profile_of_LLMs.length; i++) {
    queue[i] = new sqs.Queue(this, 'QueueS3EventFifo' + i, {
        visibilityTimeout: cdk.Duration.seconds(600),
        queueName: `queue-s3-event-for-${projectName}-${i}.fifo`,
        fifo: true,
        contentBasedDeduplication: false,
        deliveryDelay: cdk.Duration.millis(0),
        retentionPeriod: cdk.Duration.days(2),
    });
    queueUrl.push(queue[i].queueUrl);
}

// Lambda for s3 event manager
const lambdaS3eventManager = new lambda.Function(this, `lambda-s3-event-manager-for-${projectName}`, {
    description: 'lambda for s3 event manager',
    functionName: `lambda-s3-event-manager-for-${projectName}`,
    handler: 'lambda_function.lambda_handler',
    runtime: lambda.Runtime.PYTHON_3_11,
    code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-s3-event-manager')),
    timeout: cdk.Duration.seconds(60),
    environment: {
        sqsFifoUrl: JSON.stringify(queueUrl),
        nqueue: String(profile_of_LLMs.length)
    }
});
for (let i = 0; i < profile_of_LLMs.length; i++) {
    queue[i].grantSendMessages(lambdaS3eventManager); // permision for SQS putItem
}
```

S3의 이벤트를 저장한 SQS에서 문서 정보를 가져와서 RAG에 등록합니다. 

```typescript
// Lambda for document manager
let lambdDocumentManager: any[] = [];
for (let i = 0; i < profile_of_LLMs.length; i++) {
    lambdDocumentManager[i] = new lambda.DockerImageFunction(this, `lambda-document-manager-for-${projectName}-${i}`, {
        description: 'S3 document manager',
        functionName: `lambda-document-manager-for-${projectName}-${i}`,
        role: roleLambdaWebsocket,
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-document-manager')),
        timeout: cdk.Duration.seconds(600),
        memorySize: 8192,
        environment: {
            bedrock_region: profile_of_LLMs[i].bedrock_region,
            s3_bucket: s3Bucket.bucketName,
            s3_prefix: s3_prefix,
            opensearch_account: opensearch_account,
            opensearch_passwd: opensearch_passwd,
            opensearch_url: opensearch_url,
            roleArn: roleLambdaWebsocket.roleArn,
            path: 'https://' + distribution.domainName + '/',
            sqsUrl: queueUrl[i],
            max_object_size: String(max_object_size),
            supportedFormat: supportedFormat,
            profile_of_LLMs: JSON.stringify(profile_of_LLMs),
            enableParallelSummay: enableParallelSummay
        }
    });
    s3Bucket.grantReadWrite(lambdDocumentManager[i]); // permission for s3
    lambdDocumentManager[i].addEventSource(new SqsEventSource(queue[i])); // permission for SQS
}

// s3 event source
const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
    events: [
        s3.EventType.OBJECT_CREATED_PUT,
        s3.EventType.OBJECT_REMOVED_DELETE
    ],
    filters: [
        { prefix: s3_prefix + '/' },
    ]
});
lambdaS3eventManager.addEventSource(s3PutEventSource); 
```

Client에서 Websocket의 접속정보를 가져오기 위한 Lambda를 설정합니다. 

```typescript
// lambda - provisioning
const lambdaProvisioning = new lambda.Function(this, `lambda-provisioning-for-${projectName}`, {
    description: 'lambda to earn provisioning info',
    functionName: `lambda-provisioning-api-${projectName}`,
    handler: 'lambda_function.lambda_handler',
    runtime: lambda.Runtime.PYTHON_3_11,
    code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-provisioning')),
    timeout: cdk.Duration.seconds(30),
    environment: {
        wss_url: wss_url,
    }
});

// POST method - provisioning
const provisioning_info = api.root.addResource("provisioning");
provisioning_info.addMethod('POST', new apiGateway.LambdaIntegration(lambdaProvisioning, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for provisioning api
distribution.addBehavior("/provisioning", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

API Gateway를 배포합니다.

```typescript
// deploy components
new componentDeployment(scope, `component-deployment-of-${projectName}`, websocketapi.attrApiId)


export class componentDeployment extends cdk.Stack {
    constructor(scope: Construct, id: string, appId: string, props?: cdk.StackProps) {
        super(scope, id, props);

        new apigatewayv2.CfnDeployment(this, `api-deployment-of-${projectName}`, {
            apiId: appId,
            description: "deploy api gateway using websocker",  // $default
            stageName: stage
        });
    }
} 
```
