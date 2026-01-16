Place DINOv3 `.pth` weights in this folder for local runs.

Current expected file (from your machine):
- `models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`

Airflow/DAG usage:
- Set `DINO_MODEL_HOST_PATH=D:/rasterpipeline/models/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
  so the `compute_embeddings` task bind-mounts the folder and passes `--model_path`.
