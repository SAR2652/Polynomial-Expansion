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

# python -m jax_implementation.ddp_train \
#     --input_filepath /kaggle/input/polynomialexpansion/training.csv \
#     --output_dir /kaggle/working/output \
#     --tokenizer_filepath /kaggle/input/polynomialexpansion/tokenizer.joblib \
#     --hidden_dim 64 \
#     --embed_dim 64 \
#     --epochs 225 \
#     --batch_size 1536 \
#     --learning_rate 1e-4 \
#     --bidirectional

python -m jax_implementation.train \
    --input_filepath /kaggle/input/polynomialexpansion/training.csv \
    --output_dir /kaggle/working/output \
    --tokenizer_filepath /kaggle/input/polynomialexpansion/tokenizer.joblib \
    --hidden_dim 64 \
    --embed_dim 64 \
    --epochs 225 \
    --batch_size 768 \
    --learning_rate 1e-4 \
    --bidirectional
