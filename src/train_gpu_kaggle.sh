# python -m pytorch_new_implementation.train \
#     --input_filepath /kaggle/input/polynomialexpansion/training.csv \
#     --output_dir /kaggle/working/output \
#     --tokenizer_filepath /kaggle/input/polynomialexpansion/tokenizer.joblib \
#     --hidden_dim 64 \
#     --embed_dim 64 \
#     --epochs 1000 \
#     --batch_size 768 \
#     --learning_rate 1e-4 \
#     --bidirectional

python -m src.jax_implementation.ddp_train \
    --input_dir /kaggle/input/polynomialexpansion \
    --output_dir /kaggle/working/output \
    --tokenizer_filepath /kaggle/input/polynomialexpansion/tokenizer.joblib \
    --hidden_dim 64 \
    --embed_dim 64 \
    --epochs 125 \
    --batch_size 1536 \
    --learning_rate 2e-4 \
    --early_stopping_patience 10 \
    --bidirectional \
    --profile \
    --disable_wandb \
    --use_cache \
    --ddp \
    --continue_from_ckpt \
    --old_ckpt_dir ddp_kvc_results/output/checkpoints
