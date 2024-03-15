#!/bin/sh
if [ ! -f ./cache/slot/embeddings.pt  ] || [ ! -f ./cache/slot/tag2idx.json  ] || [ ! -f ./cache/slot/vocab.pkl  ]; then
    echo "Can't find corresponding cache files. Preparing to preprocess."
    if [ ! -f glove.840B.300d.txt ]; then
        echo "Can't find GloVe file Start downloading."
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
        unzip glove.840B.300d.zip
    fi
    echo "Start preprocessing."
    python3 preprocess_slot.py
fi

echo "Start training."
python3 train_slot.py --data_dir="${1}" --ckpt_dir="${2}" --num_epoch=10 --max_len=128 --batch_size=128 --lr=1e-3 --hidden_size=256 --num_layers=2 --dropout=0.1 --bidirectional=True