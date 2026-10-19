# Offline Edge Model (ONNX)
#
# 1. Train: python ml/training/train.py --data ml/data --epochs 5
# 2. Export: python ml/export_edge/export_onnx.py \
#       --weights ml/training/artifacts/model.pt \
#       --metrics ml/training/artifacts/metrics.json \
#       --output ml/export_edge/artifacts
# 3. Infer offline: python ml/export_edge/infer_onnx.py \
#       --image leaf.jpg \
#       --model ml/export_edge/artifacts/classifier.onnx \
#       --classes ml/export_edge/artifacts/classes.json
#
# Install: pip install -r ml/export_edge/requirements.txt
