# Outbreak Dashboard (Athena + QuickSight)
#
# ARCHIVED: this dashboard needs the AWS analytics stack, which is no
# longer deployed. The infrastructure is in archive/aws/terraform/analytics/
# and the data export in archive/aws/ml/analytics/export_to_s3.py.
# See archive/aws/README.md for how to bring it back.
#
# Infrastructure: archive/aws/terraform/analytics/
#   - S3 bucket for case data + Athena results
#   - Glue database + crawler (cases table)
#   - Athena workgroup
#
# Queries: docs/outbreak_analytics.sql
#
# QuickSight setup:
#   1. Create QuickSight account (Enterprise edition)
#   2. Add Athena as a data source (workgroup: crop-advisor-dev-analytics)
#   3. Select the cases table from crop_advisor_analytics database
#   4. Build visuals:
#      - Disease distribution by crop (bar chart)
#      - Outbreak hotspots by location (geospatial)
#      - Daily diagnosis trend (line chart)
#      - Confidence distribution (pie chart)
#
# Data export: python archive/aws/ml/analytics/export_to_s3.py --bucket <bucket>
