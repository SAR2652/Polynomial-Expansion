python -m pytorch_new_implementation.train \
    --input_filepath /kaggle/input/polynomialexpansion/training.csv \
    --output_dir /kaggle/working/output \
    --tokenizer_filepath /kaggle/input/polynomialexpansion/tokenizer.joblib \
    --hidden_dim 32 \
    --embed_dim 32 \
    --epochs 1000 \
    --batch_size 768 \
    --learning_rate 1e-4

# python -m jax_implementation.train \
#     --input_filepath /kaggle/input/polynomialexpansion/training.csv \
#     --model_parameters_dir /kaggle/working/output \
#     --output_dir /kaggle/working/output \
#     --tokenizer_filepath /kaggle/input/polynomialexpansion/tokenizer.joblib \
#     --hidden_size 64 \
#     --epochs 50
