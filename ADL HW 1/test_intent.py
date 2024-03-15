import json
import pickle
import string
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab, pad_to_len


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)

    test_dataloader = DataLoader(dataset, batch_size = args.batch_size, collate_fn = dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location = device))
    #print(model)

    model.eval()
    prediction = None
    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            X = torch.tensor(data['padding']).squeeze().to(device)
    
            pred = model(X).squeeze().argmax(dim=1)

            if prediction == None:
                prediction = pred
            else:
                prediction = torch.cat((prediction, pred))

    #print(prediction.size(), len(dataset))

    # TODO: write prediction to file (args.pred_file)
    write_output(dataset, prediction, args.pred_file)

    print("Predict done!")

def write_output(dataset, prediction, output_file):
    out = []
    out.append(['id', 'intent'])
    for i, data in enumerate(dataset):
        out.append([data['id'], dataset.idx2label(prediction[i].item())])

    with open(output_file, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(out)

    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
