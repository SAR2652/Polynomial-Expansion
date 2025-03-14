# python -m pytorch_new_implementation.batched_inference --bidirectional
python -m jax_implementation.batched_inference
    --tokenizer_filepath /content/tokenizer.joblib \
    --hidden_dim 64 \
    --embed_dim 64 \
    --bidirectional \
    --ckpt_dir ./best_model_saca_bidirect_217
