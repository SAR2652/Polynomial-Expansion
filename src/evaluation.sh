# python -m pytorch_new_implementation.batched_inference --bidirectional

# python -m jax_implementation.batched_inference \
#     --input_filepath /content/validation.csv \
#     --tokenizer_filepath /content/tokenizer.joblib \
#     --hidden_dim 64 \
#     --embed_dim 64 \
#     --bidirectional \
#     --ckpt_dir /content/checkpoints


python -m src.jax_implementation.batched_evaluation \
    --input_filepath ./output/test.csv \
    --tokenizer_filepath ./output/tokenizer.joblib \
    --hidden_dim 64 \
    --embed_dim 64 \
    --bidirectional \
    --ckpt_dir ./output/normal/output/checkpoints \
    --batch_size 128
