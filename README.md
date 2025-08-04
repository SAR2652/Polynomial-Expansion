# Polynomial-Expansion

### ⚙️ Training Performance Summary

| Config      | GPUs | KV Caching | DDP | Steps/sec (Train) | Steps/sec (Val) | Steps/sec (Overall) | Tokens/sec (Train) | Tokens/sec (Val) | Tokens/sec (Overall) | Time/Epoch (s) | Speedup vs Baseline |
|-------------|------|------------|-----|-------------------|------------------|----------------------|---------------------|-------------------|------------------------|----------------|----------------------|
| Baseline    | 1    | ❌         | ❌  | 8.85              | 27.49            | 19.16                | 151,436             | 644,328           | 424,100                | 254.69         | –                    |
| DDP only    | 1    | ❌         | ✅  | 11.84             | 28.19            | 20.89                | 235,186             | 661,566           | 471,056                | 194.32         | 23.70%               |
| KV only     | 2    | ✅         | ❌  | 11.48             | 31.58            | 22.60                | 227,866             | 739,890           | 511,114                | 188.46         | 25.80%               |
| DDP + KV    | 2    | ✅         | ✅  | 15.63             | 34.96            | 26.32                | 333,705             | 820,762           | 603,141                | 151.91         | 40.30%               |


### Archived

The model was trained for 22 epochs and had a final training loss of 185.99 over 800,000 samples trained in batch sizes of 16 selected at random (seed = 42) from the file train.txt which contains the training data of 1,000,000 samples. It took nearly 19 hours and 37 minutes to train the model in order to achieve this performance. The model achieves  The model currently achieves an accuracy of 98.74% on the entire data set (1,000,000 samples) and 98.35% on validation data which consist of the remaining 200,000 samples that were not used for training. (Check [Polynomial_Training_and_Evaluation.ipynb](https://github.com/SAR2652/Polynomial-Expansion/blob/main/Polynomial_Training_and_Evaluation.ipynb)). Each epoch takes approximately 54 minutes to complete with GPU Support. Inference over 1,000,000 samples takes nearly 144 minutes with GPU Support due to the complexity of the model since it involves a <b>for</b> loop that iterates over the maximum target length for each of the 1,000,000 samples while generating the next token.  The model comprises of an LSTM based Encoder and a Decoder that performs cross attention between the encoder of the outputs and the hidden states. The context vector is created by taking the Einstein Sum of the attention output and the encoder outputs and is concatenated with the target embedding and passed as input to an LSTM after which the LSTM output is passed to a fully connected layer to obtain the final prediction.

## References
The model architecture was inspired from the following tutorial.<br>
**Link:** https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq_attention/seq2seq_attention.py
