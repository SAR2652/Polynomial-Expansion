python -m pytorch_new_implementation.train \
    --input_filepath /content/drive/MyDrive/polynomial_expansion/output/training.csv \
    --output_dir /content/drive/MyDrive/polynomial_expansion/output/ \
    --tokenizer_filepath /content/drive/MyDrive/polynomial_expansion/output/tokenizer.joblib \
    --hidden_dim 64 \
    --embed_dim 64 \
    --epochs 250 \
    --batch_size 768 \
    --learning_rate 1e-6 \
    --continue_from_ckpt \
    --ckpt_file /content/drive/MyDrive/polynomial_expansion/output/best_model_272.pth

# python -m jax_implementation.train --input_filepath /content/drive/MyDrive/polynomial_expansion/output/training.csv --output_dir /content/drive/MyDrive/polynomial_expansion/output --tokenizer_filepath /content/drive/MyDrive/polynomial_expansion/output/tokenizer.joblib
# python train.py ./data/training.csv --accelerator cuda --hidden_size 320 --model_path /content/drive/MyDrive/new_encoder_decoder_model.pt --tokenizer_filepath ./tokenizers/tokenizer.pickle --epochs 200