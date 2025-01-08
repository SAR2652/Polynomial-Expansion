python -m pytorch_new_implementation.train \
    --input_filepath ./output/training.csv \
    --output_dir ./output/ \
    --tokenizer_filepath ./output/tokenizer.joblib \
    --hidden_dim 64 \
    --embed_dim 64 \
    --epochs 500 \
    --batch_size 768 \
    --learning_rate 1e-3 \
    # --continue_from_ckpt \
    # --ckpt_file ./output/best_model.pth
