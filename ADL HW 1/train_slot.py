import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import TagClsDataset
from torch.utils.data import DataLoader

from model import TagClassifier

from utils import Vocab, pad_to_len

from seqeval.metrics import classification_report, accuracy_score, precision_score
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    #print(data.items())
    datasets: Dict[str, TagClsDataset] = {
        split: TagClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_dataloader = DataLoader(datasets[TRAIN], batch_size = args.batch_size, collate_fn=datasets[TRAIN].collate_fn)
    eval_dataloader = DataLoader(datasets[DEV], batch_size = args.batch_size, collate_fn=datasets[DEV].collate_fn)
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    # TODO: init model and move model to target device(cpu / gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TagClassifier(
        embeddings, 
        args.hidden_size, 
        args.num_layers, 
        args.dropout, 
        args.bidirectional, 
        datasets[TRAIN].num_classes
    ).to(device)
    print(f"Using {device} device")
    print(model)

    # TODO: init optimizer
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_model, best_acc = model, 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train(train_dataloader, model, loss_fn, optimizer, vocab, datasets[TRAIN], device)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        acc = test(eval_dataloader, model, loss_fn, vocab, datasets[DEV], device)

        if acc > best_acc:
            best_acc = acc
            best_model = model
    


    print('Best result', best_acc)
    model_path = str(args.ckpt_dir) + '/' + str(round(best_acc, 5)) + '.pt'
    print(model_path)
    torch.save(model.state_dict(), model_path)
    
    #seqeval_test(eval_dataloader, best_model, vocab, datasets[DEV], device)

def train(dataloader, model, loss_fn, optimizer, vocab, dataset, device):
    model.train()
    train_loss, correct = 0, 0

    for batch, data in enumerate(dataloader):
        X = torch.tensor(data['pad_tokens']).squeeze().to(device)
        y = torch.tensor(data['pad_tags']).to(device)

        pred = model(X)
        
        loss = loss_fn(pred, y)

        train_loss += loss
        correct += correct_num(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    #print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

    return

def test(dataloader, model, loss_fn, vocab, dataset, device):
    model.eval()
    test_loss, all_correct, partial_correct = 0, 0, 0
    
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            X = torch.tensor(data['pad_tokens']).squeeze().to(device)
            y = torch.tensor(data['pad_tags']).to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            all_correct += correct_num(pred, y)
            partial_correct += ((pred.argmax(1) == y).sum()).sum().item()
            #print(y)
            #print(pred.argmax(1))
    
    test_loss /= len(dataloader)
    #print(all_correct, partial_correct,len(dataloader.dataset))

    all_correct /= len(dataloader.dataset)    
    #print(f"Test Error: \n Accuracy: {(100*all_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
    return all_correct

def seqeval_test(dataloader, model, vocab, dataset, device):
    model.eval()
    
    prediction = None
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            X = torch.tensor(data['pad_tokens']).squeeze().to(device)
            y = torch.tensor(data['pad_tags']).to(device)

            pred = model(X).squeeze().argmax(dim=1)

            if prediction == None:
                prediction = pred
            else:
                prediction = torch.cat((prediction, pred))

    y_true = []
    y_pred = []
    for i, data in enumerate(dataset):
        pred_tags = []
        for j in range(len(data['tags'])):
            pred_tags.append(dataset.idx2tag(prediction[i][j].item()))

        y_pred.append(pred_tags)
        y_true.append(data['tags'])
    
    #print(y_true[:3], y_pred[:3])

    print(classification_report(y_true, y_pred, scheme = IOB2, mode = 'strict'))

def correct_num(pred, y):
    correct_num = 0
    for i in range(y.size(0)):
        is_correct = True
        for j, tag in enumerate(y[i]):
            #print('tag', tag.item(), 'pred', pred[i].argmax(0)[j].item())
            if tag.item() == -1:
                continue
            if tag.item() != pred[i].argmax(0)[j].item():
                is_correct = False
                break
        
        if is_correct:
            correct_num += 1
    
    return correct_num

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
