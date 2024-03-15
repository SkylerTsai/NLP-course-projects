import json
import pickle
import string
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from torch.utils.data import DataLoader

from model import SeqClassifier

from utils import Vocab, pad_to_len

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_dataloader = DataLoader(datasets[TRAIN], batch_size = args.batch_size, collate_fn = datasets[TRAIN].collate_fn)
    eval_dataloader = DataLoader(datasets[DEV], batch_size = args.batch_size, collate_fn = datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeqClassifier(embeddings, 
                args.hidden_size, 
                args.num_layers, 
                args.dropout, 
                args.bidirectional, 
                datasets[TRAIN].num_classes
            ).to(device)
    print(f"Using {device} device")
    print(model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    best_model, best_acc = model, 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        train(train_dataloader, model, loss_fn, optimizer, vocab, datasets[TRAIN], device)

        acc = test(eval_dataloader, model, loss_fn, vocab, datasets[DEV], device)

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print('Best result', best_acc)
    model_path = str(args.ckpt_dir) + '/' + str(round(best_acc, 5)) + '.pt'
    print(model_path)
    torch.save(model.state_dict(), model_path)

def train(dataloader, model, loss_fn, optimizer, vocab, dataset, device):
    train_loss, correct = 0, 0

    model.train()
    for batch, data in enumerate(dataloader):
        X = torch.tensor(data['padding']).squeeze().to(device)
        y = torch.tensor(data['label']).to(device)
        pred = model(X)

        loss = loss_fn(pred, y)
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct += (pred.argmax(1) == y).sum().item()
    
    train_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    #print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        
    return

def test(dataloader, model, loss_fn, vocab, dataset, device):
    test_loss, correct = 0, 0
    
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            X = torch.tensor(data['padding']).squeeze().to(device)
            y = torch.tensor(data['label']).to(device)
            

            pred = model(X).squeeze()
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
        
            #print(y)
            #print(pred.argmax(1))
    
    test_loss /= len(dataloader)
    #print(correct, len(dataloader.dataset))

    correct /= len(dataloader.dataset)    
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
    return correct

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
