#!/bin/bash
# ===========================================================================
# deploy.sh — One-command AWS deployment for Secure RAG System
# ===========================================================================
# Usage:
#   ./scripts/deploy.sh <environment> <openai-api-key>
#
# Example:
#   ./scripts/deploy.sh production sk-xxxxxxx
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENVIRONMENT="${1:-production}"
OPENAI_API_KEY="${2:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="secure-rag-system"
IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')"
STACK_NAME="${ENVIRONMENT}-rag-stack"

# ---------------------------------------------------------------------------
# Colors for output
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

log_info "Starting deployment to ${ENVIRONMENT} environment..."

if [[ -z "${OPENAI_API_KEY}" ]]; then
    log_error "OpenAI API key is required. Usage: ./scripts/deploy.sh <environment> <openai-api-key>"
fi

# Check AWS CLI is installed
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not found. Install it from https://aws.amazon.com/cli/"
fi

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Install it from https://www.docker.com/"
fi

# Check AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    log_error "AWS credentials not configured. Run 'aws configure' first."
fi

log_info "✓ Prerequisites validated"
log_info "  AWS Account: ${AWS_ACCOUNT_ID}"
log_info "  Region:      ${AWS_REGION}"
log_info "  Image Tag:   ${IMAGE_TAG}"

# ---------------------------------------------------------------------------
# Step 1: Create ECR repository if it doesn't exist
# ---------------------------------------------------------------------------

log_info "Step 1/5: Checking ECR repository..."

if ! aws ecr describe-repositories \
    --repository-names "${ECR_REPOSITORY}" \
    --region "${AWS_REGION}" &> /dev/null; then
    
    log_warn "ECR repository not found. Creating..."
    aws ecr create-repository \
        --repository-name "${ECR_REPOSITORY}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    log_info "✓ ECR repository created"
else
    log_info "✓ ECR repository exists"
fi

# ---------------------------------------------------------------------------
# Step 2: Build Docker image
# ---------------------------------------------------------------------------

log_info "Step 2/5: Building Docker image..."

docker build \
    -t "${ECR_REPOSITORY}:${IMAGE_TAG}" \
    -t "${ECR_REPOSITORY}:latest" \
    -f Dockerfile \
    .

log_info "✓ Docker image built: ${ECR_REPOSITORY}:${IMAGE_TAG}"

# ---------------------------------------------------------------------------
# Step 3: Push image to ECR
# ---------------------------------------------------------------------------

log_info "Step 3/5: Pushing image to ECR..."

# Login to ECR
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Tag and push
docker tag "${ECR_REPOSITORY}:${IMAGE_TAG}" \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

docker tag "${ECR_REPOSITORY}:latest" \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest"

docker push "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
docker push "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest"

log_info "✓ Image pushed to ECR"

# ---------------------------------------------------------------------------
# Step 4: Deploy CloudFormation stack
# ---------------------------------------------------------------------------

log_info "Step 4/5: Deploying CloudFormation stack..."

# Check if stack exists
if aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" &> /dev/null; then
    
    OPERATION="update"
    log_info "Stack exists. Updating..."
else
    OPERATION="create"
    log_info "Stack does not exist. Creating..."
fi

aws cloudformation ${OPERATION}-stack \
    --stack-name "${STACK_NAME}" \
    --template-body file://cloudformation.yaml \
    --parameters \
        ParameterKey=EnvironmentName,ParameterValue="${ENVIRONMENT}" \
        ParameterKey=ImageTag,ParameterValue="${IMAGE_TAG}" \
        ParameterKey=OpenAIAPIKey,ParameterValue="${OPENAI_API_KEY}" \
    --capabilities CAPABILITY_IAM \
    --region "${AWS_REGION}"

log_info "Waiting for stack ${OPERATION} to complete (this may take 5-10 minutes)..."

aws cloudformation wait stack-${OPERATION}-complete \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}"

log_info "✓ CloudFormation stack ${OPERATION}d successfully"

# ---------------------------------------------------------------------------
# Step 5: Get outputs and display
# ---------------------------------------------------------------------------

log_info "Step 5/5: Retrieving deployment information..."

ALB_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerURL'].OutputValue" \
    --output text)

ECS_CLUSTER=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='ECSClusterName'].OutputValue" \
    --output text)

ECS_SERVICE=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='ECSServiceName'].OutputValue" \
    --output text)

LOG_GROUP=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='LogGroupName'].OutputValue" \
    --output text)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo " Deployment Complete! 🎉"
echo "=========================================="
echo ""
echo "Environment:     ${ENVIRONMENT}"
echo "Image Tag:       ${IMAGE_TAG}"
echo "ALB URL:         ${ALB_URL}"
echo "ECS Cluster:     ${ECS_CLUSTER}"
echo "ECS Service:     ${ECS_SERVICE}"
echo "Log Group:       ${LOG_GROUP}"
echo ""
echo "Health Check:    ${ALB_URL}/health"
echo "API Docs:        ${ALB_URL}/docs"
echo "Metrics:         ${ALB_URL}/metrics"
echo ""
echo "To view logs:"
echo "  aws logs tail ${LOG_GROUP} --follow --region ${AWS_REGION}"
echo ""
echo "To query the API:"
echo "  curl -X POST ${ALB_URL}/query \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"What is Python?\"}'"
echo ""
echo "=========================================="