#!/bin/sh
python3 test_intent.py --test_file "${1}" --ckpt_path best_intent.pt --pred_file "${2}" --max_len=128 --batch_size=256 --hidden_size=256 --num_layers=2 --dropout=0.2 --bidirectional=True