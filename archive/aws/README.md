# Archived: the AWS deployment path

Nothing here runs any more. The project is deployed on **Render** (API)
and **Netlify** (web app), and the classifier ships as an ONNX file in
the container image, so none of this is needed.

It is kept in the repository (not deleted) because the history is still
there and a future scale-up might want it back.

| What | Why it was archived |
|---|---|
| `terraform/` | ECS + ECR + ALB + VPC + NAT, a SageMaker endpoint, a Bedrock guardrail, Athena + Glue, DynamoDB and S3. Render replaces the compute, and the ONNX model replaces SageMaker. |
| `ml/pipelines/training_pipeline.py` | A SageMaker Pipeline that trains the model on AWS. The local trainer (`ml/training/train.py`) and the Kaggle path cover training. |
| `ml/pipelines/train_sagemaker.py` | The entry point the SageMaker pipeline runs inside its container. |
| `ml/analytics/export_to_s3.py` | Exports the review cases to S3 for Athena. The `apps/expert-review` data stays local. |
| `workflows/ml-pipeline.yml` | The GitHub Actions workflow that ran the pipeline above. |

## Bringing it back

```bash
git mv archive/aws/terraform infra/terraform
git mv archive/aws/ml/pipelines ml/pipelines
git mv archive/aws/ml/analytics ml/analytics
git mv archive/aws/workflows/ml-pipeline.yml .github/workflows/ml-pipeline.yml
```

The AWS paths in the application code are still in place and still
guarded by environment variables, so nothing has to be rewritten to use
them again:

- `CA_SAGEMAKER_ENDPOINT` - classify with a SageMaker endpoint instead
  of the bundled ONNX model
- `CA_GUARDRAIL_ID` - screen the advice text with a Bedrock guardrail
- `CA_SARVAM_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS` - speech
- `boto3` is only imported inside the functions that need it, so the
  image does not carry the AWS SDK unless those paths are used.
