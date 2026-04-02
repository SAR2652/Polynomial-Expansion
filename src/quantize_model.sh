python -m src.jax_implementation.quantize_model_weights_calibrated \
    --calib_data_path ./output/validation.csv \
    --num_calib_samples 512 \
    --calib_batch_size 64 \
    --bidirectional
