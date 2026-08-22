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


python -m src.jax_implementation.ddp_train_lr_schedule \
    --input_dir /kaggle/input/polynomialexpansion \
    --output_dir /kaggle/working/output \
    --tokenizer_filepath /kaggle/input/polynomialexpansion/tokenizer.joblib \
    --hidden_dim 128 \
    --embed_dim 128 \
    --epochs 500 \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --bidirectional \
    --disable_wandb \
    --use_cache \
    --profile \
    --ddp


# --continue_from_ckpt \
# --old_ckpt_dir ddp_kvc_results/checkpoints

# python -m src.jax_implementation.ddp_train_lr_schedule \
#     --input_dir ./output \
#     --output_dir ./output \
#     --tokenizer_filepath ./output/tokenizer.joblib \
#     --hidden_dim 64 \
#     --embed_dim 64 \
#     --epochs 500 \
#     --batch_size 1024 \
#     --learning_rate 1e-4 \
#     --bidirectional \
#     --disable_wandb \
#     --use_cache \
#     --profile