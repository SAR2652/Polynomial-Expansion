# Polynomial-Expansion

### ⚙️ Training Performance Summary

| Config      | GPUs | KV Caching | DDP | Steps/sec (Train) | Steps/sec (Val) | Steps/sec (Overall) | Tokens/sec (Train) | Tokens/sec (Val) | Tokens/sec (Overall) | Time/Epoch (s) | Speedup vs Baseline |
|-------------|------|------------|-----|-------------------|------------------|----------------------|---------------------|-------------------|------------------------|----------------|----------------------|
| Baseline    | 1    | ❌         | ❌  | 8.50              | 24.32            | 17.25                | 141,579             | 568,928           | 377,985                | 262.14         | –                    |
| DDP only    | 1    | ❌         | ✅  | 11.02             | 25.46            | 19.01                | 213,771             | 596,187           | 425,320                | 203.68         | 22.30%               |
| KV only     | 2    | ✅         | ❌  | 10.69             | 27.22            | 19.83                | 207,358             | 636,255           | 444,620                | 196.72         | 24.96%               |
| DDP + KV    | 2    | ✅         | ✅  | 14.09             | 32.06            | 24.03                | 295,054             | 751,456           | 547,532                | 162.11         | 38.16%               |


## CUTLASS implementation notes:
1. Quantized weights to INT8 and bias to INT32.
2. Started implementation with CUTLASS 4.1. It has no support for INT8 kernels
3. Downgraded to CUTLASS 3.2.2. It cannot do INT8 for my case because:
```
CUTLASS 3.2.2 simply does not have a valid warp‑level MMA kernel for:

int8_t × int8_t → int32_t

RowMajor × RowMajor

OpClassTensorOp

ArchTag = Sm80 or Sm89 or anything except Sm90

CUTLASS 3.x moved most int8 kernels into the SM90 GMMA/TMA path, and the older SM80/SM89 int8 kernels were not fully ported.

That’s why the iterator type is incomplete — CUTLASS never defined it.

Earlier attempts all failed because:
1. SM90 + TMA kernels
These only work on Hopper (H100).
RTX 5070 cannot run them → incomplete type.

2. SM100
CUTLASS 3.2.2 does not define arch::Sm100 yet → compile error.

3. SM80 + device::GemmUniversal
CUTLASS 3.2.2 does not ship an int8 TensorOp kernel for SM80 in the new 3.x API.
The warp‑level iterator fails → incomplete type.

4. device::Gemm
CUTLASS 3.x removed device::Gemm entirely → namespace error.
```

3. Requantizing everything to BF16 instead of FP16 and retrying
```
For RTX 5070 / SM89 / int8→int32 GEMM:

Do not use collective::CollectiveBuilder — it’s not implemented.
```
4. Matrix MUltiplication now works for testing Linear layer using the device::GEMM API however, implementation is incorrect since linear layer takes s input dimension 128 but input shape of embedding dimension is 64. Despite of this issue CUTLASS works which means it has no input validation capabilities. Need to implement this stuff manually.


## Archived

The model was trained for 22 epochs and had a final training loss of 185.99 over 800,000 samples trained in batch sizes of 16 selected at random (seed = 42) from the file train.txt which contains the training data of 1,000,000 samples. It took nearly 19 hours and 37 minutes to train the model in order to achieve this performance. The model achieves  The model currently achieves an accuracy of 98.74% on the entire data set (1,000,000 samples) and 98.35% on validation data which consist of the remaining 200,000 samples that were not used for training. (Check [Polynomial_Training_and_Evaluation.ipynb](https://github.com/SAR2652/Polynomial-Expansion/blob/main/Polynomial_Training_and_Evaluation.ipynb)). Each epoch takes approximately 54 minutes to complete with GPU Support. Inference over 1,000,000 samples takes nearly 144 minutes with GPU Support due to the complexity of the model since it involves a <b>for</b> loop that iterates over the maximum target length for each of the 1,000,000 samples while generating the next token.  The model comprises of an LSTM based Encoder and a Decoder that performs cross attention between the encoder of the outputs and the hidden states. The context vector is created by taking the Einstein Sum of the attention output and the encoder outputs and is concatenated with the target embedding and passed as input to an LSTM after which the LSTM output is passed to a fully connected layer to obtain the final prediction.

## References
The model architecture was inspired from the following tutorial.<br>
**Link:** https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq_attention/seq2seq_attention.py
