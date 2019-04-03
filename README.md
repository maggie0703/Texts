# POS Tagging
Data From NLTK Brown corpora

Summary:

1.Preprocess_data.py: load data, split data, create word embedding and one hot encoding for tags 

2.Config_.py: define parameters

3.Train.py:t rain model and saved model

4.Test.py: apply model to test data

5.Input.py:input a sentence to test to obtain output

6.lstm.py, bilstm.py, lstmcrf, bilstmcrf.py: 4 models contains: LSTM model, BiLSTM model, LSTM_CRF. amd BiLSTM_CRF

Results:
1. test accruacy of LSTM:0.90
2. test accuracy of BiLSTM： 0.92
3. test accuracy of LSTMCRF: 0.94
4. test accuracy of BiLSTMCRF:0.95
Conclusion: BiLSTM+CRF outperforms than other three models

