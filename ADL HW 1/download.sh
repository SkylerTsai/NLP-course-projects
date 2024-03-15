#!/bin/sh
echo "Cache downloading..."
wget https://www.dropbox.com/s/4ivbcl73m7mnwg5/cache.zip?dl=1 -O cache.zip
unzip cache.zip
echo "Intent model downloading..."
wget https://www.dropbox.com/s/16kvkq5d0vrsakt/best_intent.pt?dl=1 -O best_intent.pt
echo "Slot model downloading..."
wget https://www.dropbox.com/s/yvnuxxqh4lublq0/best_slot.pt?dl=0 -O best_slot.pt
