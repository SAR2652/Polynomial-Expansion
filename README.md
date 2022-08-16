# Polynomial-Expansion

The model was trained for 22 epochs and had a final training loss of 185.99 over 800,000 samples selected at random (seed = 42) from the file train.txt which contains the training data of 1,000,000 samples. It took nearly 19 hours and 37 minutes to train the model in order to achieve this performance. The model currently achieves an accuracy of 98.35% on validation data which consist of the remaining 200,000 samples that were not used for training. Each epoch takes approximately 54 minutes to complete. The model comprises of an LSTM based Encoder and a Decoder that performs cross attention between the encoder of the outputs and the hidden states. The context vector is created by taking the einstein sum of the attention output and the encoder outputs and is concatenated with the target embedding and passed as input to an LSTM after which the LSTM output is passed to a fully connected layer to obtain the final prediction.

## References
The model architecture was inspired from the following tutorial.<br>
**Link:** https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/Seq2Seq_attention/seq2seq_attention.py
