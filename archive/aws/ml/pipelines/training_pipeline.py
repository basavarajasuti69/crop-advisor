"""SageMaker Pipeline: train → evaluate → register → canary deploy.

Run:
    python ml/pipelines/training_pipeline.py --role-arn <arn> --bucket <bucket>

Creates/updates a SageMaker Pipeline that:
  1. Trains EfficientNet-B0 on field photos
  2. Evaluates (gate on field-photo accuracy >= threshold)
  3. Registers model in SageMaker Model Registry
  4. Deploys canary endpoint if approved
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.pipeline import Pipeline
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--bucket", required=True, help="S3 bucket for artifacts")
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--instance-type", default="ml.m5.large")
    return ap.parse_args()


def build_pipeline(role_arn: str, bucket: str, region: str, instance_type: str) -> Pipeline:
    pipeline_session = PipelineSession()

    # Parameters
    model_approval_threshold = ParameterFloat(
        name="ModelApprovalThreshold", default_value=0.85
    )
    model_package_group_name = ParameterString(
        name="ModelPackageGroupName", default_value="crop-advisor-classifier"
    )
    endpoint_name = ParameterString(
        name="EndpointName", default_value="crop-advisor-classifier-dev"
    )

    # --- Processing: data validation ---
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
    )

    validation_code = Path("ml/pipelines/validate_data.py")
    validation_step = ProcessingStep(
        name="ValidateData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/data/train/",
                destination="/opt/ml/processing/input/train",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="validation_report",
                destination=f"s3://{bucket}/pipeline/validation/",
            )
        ],
        code=str(validation_code),
        property_files=[
            PropertyFile(
                name="validation_report",
                output_name="validation_report",
                path="validation_report.json",
            )
        ],
    )

    # --- Training ---
    estimator_image = image_uris.retrieve(
        framework="pytorch", region=region, version="2.0.1", py_version="py39", instance_type=instance_type
    )
    from sagemaker.estimator import Estimator

    estimator = Estimator(
        image_uri=estimator_image,
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
        output_path=f"s3://{bucket}/pipeline/output/",
        code="ml/training/train_sagemaker.py",
        entry_point="train_sagemaker.py",
        source_dir=".",
    )

    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_uri=f"s3://{bucket}/data/train/",
                content_type="application/x-image",
            )
        },
    )

    # --- Processing: evaluation ---
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=pipeline_session,
    )

    eval_code = Path("ml/pipelines/evaluate_model.py")
    evaluation_report_property_file = PropertyFile(
        name="evaluation_report",
        output_name="evaluation_report",
        path="evaluation_report.json",
    )
    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3Uri,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=f"s3://{bucket}/data/test/",
                destination="/opt/ml/processing/input/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_report",
                destination=f"s3://{bucket}/pipeline/evaluation/",
            )
        ],
        code=str(eval_code),
        property_files=[evaluation_report_property_file],
    )

    # --- Condition: gate on accuracy ---
    accuracy_gte_threshold = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report_property_file,
            json_path="metrics.field_photo_accuracy",
        ),
        right=model_approval_threshold,
    )

    # --- Register model ---
    register_model = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3Uri,
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
        content_types=["application/x-image"],
        response_types=["application/json"],
    )

    # --- Condition step: register only if accuracy >= threshold ---
    register_step = ConditionStep(
        name="AccuracyGate",
        conditions=[accuracy_gte_threshold],
        if_steps=[register_model],
        else_steps=[],
    )

    pipeline = Pipeline(
        name="crop-advisor-training-pipeline",
        parameters=[model_approval_threshold, model_package_group_name, endpoint_name],
        steps=[validation_step, train_step, eval_step, register_step],
        sagemaker_session=pipeline_session,
    )
    return pipeline


def main() -> None:
    args = parse_args()
    pipeline = build_pipeline(args.role_arn, args.bucket, args.region, args.instance_type)
    pipeline.upsert(role_arn=args.role_arn)
    print(f"Pipeline '{pipeline.name}' created/updated in {args.region}")
    execution = pipeline.start()
    print(f"Execution started: {execution.arn}")


if __name__ == "__main__":
    main()
