# Bedrock Guardrails for crop-disease advisor safety.
resource "aws_bedrock_guardrail" "crop_advisor_safety" {
  name                      = "${local.name_prefix}-safety-guardrail"
  description               = "Safety guardrails for crop disease advisor"
  blocked_input_messaging   = "Please ask a crop or farming related question."
  blocked_outputs_messaging = "I can't provide that advice. Please contact a local agriculture officer."

  topic_policy_config {
    topics_config {
      name       = "UnsafeChemicalDosing"
      definition = "Requests for pesticide or fungicide dosages that are not grounded in the source knowledge base"
      examples   = ["How much pesticide should I use without checking its label?"]
      type       = "DENY"
    }

    topics_config {
      name       = "OffTopicQueries"
      definition = "Queries unrelated to crops, farming, agriculture, or weather"
      examples   = ["Write me a movie review"]
      type       = "DENY"
    }
  }

  content_policy_config {
    filters_config {
      input_strength  = "HIGH"
      output_strength = "HIGH"
      type            = "MISCONDUCT"
    }
  }

  sensitive_information_policy_config {
    pii_entities_config {
      type   = "EMAIL"
      action = "ANONYMIZE"
    }

    pii_entities_config {
      type   = "PHONE"
      action = "ANONYMIZE"
    }
  }

  tags = local.tags
}

output "guardrail_id" {
  value = aws_bedrock_guardrail.crop_advisor_safety.guardrail_id
}

output "guardrail_arn" {
  value = aws_bedrock_guardrail.crop_advisor_safety.guardrail_arn
}
