# Architecture

## Region
ap-south-1 (Mumbai) — closest to Indian farmer users.

## Data flow (Phase 1)
1. Farmer uploads a leaf photo via the Next.js PWA or Telegram bot.
2. API Gateway → FastAPI on ECS Fargate.
3. LangGraph: quality check → classifier (SageMaker endpoint) → RAG → advice → critic.
4. Response returns diagnosis, grounded advice, citations, confidence, safe flag.
5. Case stored in DynamoDB; image in S3.

## Security
- Cognito user pools for auth.
- WAF on API Gateway.
- Secrets Manager for DB/API keys; KMS for encryption at rest.
- VPC endpoints for S3, DynamoDB, SageMaker (no NAT where possible).

## Cost controls
- CloudWatch budgets with alerts on day one.
- Delete idle SageMaker endpoints; use Fargate spot where possible.
