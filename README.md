# bertVersusAll

# For Using the Traditional Embed-Encode-Attend-Predict Model


# For Using the Bert Model (Minimal BERT)
1. Install: pip install -U bert-serving-server bert-serving-client
    - The server MUST be running on Python >= 3.5 with Tensorflow >= 1.10
    - Download a bert model:
      SMALL = https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
      LARGE = https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
    - Unzip to a directory to point to later (Suggested inside venv folder)


2. Install other needed libraries and download data
    - pip install keras_metrics keras pandas numpy sklearn

3. Serve Model:
    - bert-serving-start -model_dir ~/bert/models/cased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len=250
        num_workers can be used to increase num_workers (Model is loaded n-times)
        max_seq_len specifies the largest entry size to the bert model (suggest 250 for concept work)
