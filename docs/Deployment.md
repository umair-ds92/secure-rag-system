# Secure RAG System - AWS Deployment Guide

This guide walks you through deploying the Secure RAG System to AWS ECS Fargate with full production infrastructure.

---

## Architecture Overview

The deployment creates:

- **VPC** with public and private subnets across 2 AZs
- **Application Load Balancer (ALB)** for HTTP/HTTPS traffic
- **ECS Fargate Cluster** with auto-scaling (1-10 tasks)
- **ECS Task Definition** with 2 containers:
  - `rag-app` — FastAPI application
  - `chromadb` — Vector database
- **CloudWatch Logs** with 7-day retention
- **Secrets Manager** for OpenAI API key
- **Auto Scaling** based on CPU utilization (70% target)

**Estimated Monthly Cost:** $50-150 (depending on traffic)

---

## Prerequisites

### 1. AWS Account & CLI

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure
```

Required IAM permissions:
- `ec2:*` (VPC, subnets, security groups)
- `ecs:*` (cluster, service, task definition)
- `elasticloadbalancing:*` (ALB, target groups)
- `logs:*` (CloudWatch Logs)
- `secretsmanager:*` (API key storage)
- `ecr:*` (Docker image registry)
- `cloudformation:*` (stack management)

### 2. Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### 3. OpenAI API Key

Get your API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

---

## Deployment Steps

### Step 1: Clone and Configure

```bash
git clone <your-repo-url>
cd secure-rag-system

# Verify all files are present
ls -la cloudformation.yaml scripts/deploy.sh Dockerfile
```

### Step 2: Set AWS Region (Optional)

```bash
export AWS_REGION=us-east-1  # Default: us-east-1
```

### Step 3: Run Deployment Script

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh production <YOUR_OPENAI_API_KEY>
```

The script will:
1. ✓ Validate prerequisites
2. ✓ Create ECR repository
3. ✓ Build Docker image
4. ✓ Push to ECR
5. ✓ Deploy CloudFormation stack
6. ✓ Wait for stack completion (~5-10 min)
7. ✓ Display deployment info

### Step 4: Verify Deployment

```bash
# Get the ALB URL from the script output
ALB_URL=<ALB-DNS-NAME-FROM-OUTPUT>

# Test health check
curl $ALB_URL/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "chromadb_connected": true,
#   "model_loaded": true
# }

# Test query endpoint
curl -X POST $ALB_URL/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

---

## Manual Deployment (Alternative)

If you prefer manual control:

### 1. Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name secure-rag-system \
  --region us-east-1
```

### 2. Build and Push Image

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

# Build
docker build -t secure-rag-system:latest .

# Tag
docker tag secure-rag-system:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/secure-rag-system:latest

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Push
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/secure-rag-system:latest
```

### 3. Deploy CloudFormation Stack

```bash
aws cloudformation create-stack \
  --stack-name production-rag-stack \
  --template-body file://cloudformation.yaml \
  --parameters \
      ParameterKey=EnvironmentName,ParameterValue=production \
      ParameterKey=ImageTag,ParameterValue=latest \
      ParameterKey=OpenAIAPIKey,ParameterValue=<YOUR_API_KEY> \
  --capabilities CAPABILITY_IAM \
  --region us-east-1

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name production-rag-stack \
  --region us-east-1
```

---

## Monitoring

### CloudWatch Logs

View application logs:

```bash
# Get log group name
LOG_GROUP=$(aws cloudformation describe-stacks \
  --stack-name production-rag-stack \
  --query "Stacks[0].Outputs[?OutputKey=='LogGroupName'].OutputValue" \
  --output text)

# Tail logs
aws logs tail $LOG_GROUP --follow
```

### Container Insights

ECS Container Insights is enabled by default. View metrics in the CloudWatch console:

1. Navigate to **CloudWatch → Container Insights**
2. Select your cluster: `production-rag-cluster`
3. View CPU, memory, network metrics

### Prometheus Metrics

Scrape the `/metrics` endpoint with Prometheus:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['<ALB_DNS_NAME>:80']
    metrics_path: '/metrics'
```

---

## Scaling

### Manual Scaling

```bash
# Scale to 5 tasks
aws ecs update-service \
  --cluster production-rag-cluster \
  --service production-rag-service \
  --desired-count 5
```

### Auto-Scaling (Already Configured)

The stack includes auto-scaling based on CPU:
- **Target:** 70% CPU utilization
- **Min:** 1 task
- **Max:** 10 tasks
- **Scale-out cooldown:** 60 seconds
- **Scale-in cooldown:** 300 seconds

Adjust in `cloudformation.yaml` under `AutoScalingPolicy`.

---

## Updating the Deployment

### Update Application Code

```bash
# Make code changes, commit
git add .
git commit -m "Update RAG logic"
git push

# Redeploy
./scripts/deploy.sh production <YOUR_OPENAI_API_KEY>
```

The script will:
- Build a new Docker image (tagged with git commit hash)
- Push to ECR
- Update the CloudFormation stack
- ECS will perform a rolling update (zero downtime)

### Update CloudFormation Parameters

```bash
# Increase desired task count
aws cloudformation update-stack \
  --stack-name production-rag-stack \
  --use-previous-template \
  --parameters \
      ParameterKey=EnvironmentName,UsePreviousValue=true \
      ParameterKey=ImageTag,UsePreviousValue=true \
      ParameterKey=OpenAIAPIKey,UsePreviousValue=true \
      ParameterKey=DesiredCount,ParameterValue=5 \
  --capabilities CAPABILITY_IAM
```

---

## Troubleshooting

### Service Won't Start

**Symptom:** Tasks keep restarting

**Fix:**

```bash
# Check task status
aws ecs describe-services \
  --cluster production-rag-cluster \
  --services production-rag-service

# Get task ARN
TASK_ARN=$(aws ecs list-tasks \
  --cluster production-rag-cluster \
  --service-name production-rag-service \
  --query 'taskArns[0]' --output text)

# Describe task
aws ecs describe-tasks \
  --cluster production-rag-cluster \
  --tasks $TASK_ARN

# Check logs for the specific task
aws logs tail /ecs/production-rag --follow
```

Common issues:
- **OpenAI API key missing/invalid** → Update secret in Secrets Manager
- **ChromaDB port conflict** → Both containers use 8000, ensure proper networking
- **Out of memory** → Increase `TaskMemory` parameter

### Health Check Failing

**Symptom:** ALB marks tasks as unhealthy

**Fix:**

```bash
# Check ALB target group
aws elbv2 describe-target-health \
  --target-group-arn <TARGET_GROUP_ARN>

# SSH into a task (if needed)
aws ecs execute-command \
  --cluster production-rag-cluster \
  --task $TASK_ARN \
  --container rag-app \
  --interactive \
  --command "/bin/bash"
```

### High Latency

**Symptom:** Queries take >5 seconds

**Possible causes:**
- **Cold start** — First request after deployment loads the model
- **OpenAI rate limits** — Check API quota
- **Under-provisioned** — Increase CPU/memory or task count

---

## Security Hardening

### 1. Enable HTTPS

Add an SSL certificate to the ALB:

```bash
# Request certificate from ACM
aws acm request-certificate \
  --domain-name rag.example.com \
  --validation-method DNS

# Update ALB listener to use HTTPS (port 443)
# See cloudformation.yaml - add HTTPS listener with certificate ARN
```

### 2. Restrict ALB Access

Update `ALBSecurityGroup` in `cloudformation.yaml`:

```yaml
SecurityGroupIngress:
  - IpProtocol: tcp
    FromPort: 443
    ToPort: 443
    CidrIp: 10.0.0.0/8  # Internal only
```

### 3. Enable WAF

```bash
# Create WAF Web ACL with rate limiting
aws wafv2 create-web-acl \
  --name rag-waf \
  --scope REGIONAL \
  --default-action Allow={} \
  --rules file://waf-rules.json

# Associate with ALB
aws wafv2 associate-web-acl \
  --web-acl-arn <WEB_ACL_ARN> \
  --resource-arn <ALB_ARN>
```

### 4. Private Subnets

Move ECS tasks to private subnets and add NAT Gateway:

```yaml
ECSService:
  NetworkConfiguration:
    AwsvpcConfiguration:
      AssignPublicIp: DISABLED  # Change from ENABLED
      Subnets:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
```

---

## Cost Optimization

### 1. Use Spot Instances

Switch to Fargate Spot (70% cheaper):

```yaml
ECSService:
  CapacityProviderStrategy:
    - CapacityProvider: FARGATE_SPOT
      Weight: 1
```

### 2. Reduce Log Retention

```yaml
LogGroup:
  RetentionInDays: 1  # Change from 7
```

### 3. Right-Size Tasks

Monitor CPU/memory and adjust:

```bash
# If consistently <50% CPU:
TaskCpu: '512'     # Down from 1024
TaskMemory: '1024' # Down from 2048
```

---

## Cleanup

To delete all resources:

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack \
  --stack-name production-rag-stack

# Wait for deletion
aws cloudformation wait stack-delete-complete \
  --stack-name production-rag-stack

# Delete ECR images (optional)
aws ecr delete-repository \
  --repository-name secure-rag-system \
  --force
```

---

## Next Steps

1. **Ingest Documents** — Use the ingestion pipeline to populate ChromaDB
2. **Integrate Real LLM** — Replace placeholder in `main.py` with OpenAI client
3. **Add Authentication** — Implement API key or OAuth middleware
4. **Set Up CI/CD** — Automate deployments with GitHub Actions

For API usage, see [API.md](API.md).