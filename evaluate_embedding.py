import os
import csv
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_embedding_from_ckpt(filepath):
    return torch.load(filepath)["embed.weight"]


class AuthorRoleClassificationDataset(Dataset):
    def __init__(self, dirpath, is_valid, seed=0):
        with open(os.path.join(dirpath, "nodes.csv"), 'r', newline='', encoding='utf8') as f:
            reader = csv.reader(f)
            node_mapping = {row[1]: int(row[0]) for row in reader}

        with open(os.path.join(dirpath, "edges.csv"), 'r', newline='', encoding='utf8') as f:
            reader = csv.reader(f)
            self.data = [(node_mapping[row[0]], node_mapping[row[1]], int(row[2])) for row in reader]
            random.seed(seed)
            random.shuffle(self.data)
            if is_valid:
                self.data = self.data[:200]
            else:
                self.data = self.data[200:]
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class LinearModel(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        _, n_dims = embeddings.shape
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.w = nn.Linear(2*n_dims, 2)

    def forward(self, a, p):
        return self.w(torch.cat([self.embed(a), self.embed(p)], dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dims", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--embedding_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = AuthorRoleClassificationDataset(os.path.join("prep"), is_valid=False)
    valid_dataset = AuthorRoleClassificationDataset(os.path.join("prep"), is_valid=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    embeddings = get_embedding_from_ckpt(args.embedding_path)
    model = LinearModel(embeddings) 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    writer = SummaryWriter()
    writer.add_embedding(embeddings)
    step = 0
    for epoch in range(args.epochs):
        with tqdm(train_loader, desc="Epoch %d" % epoch, ncols=100) as train_tbar:
            for batch in train_tbar:
                author, paper, label = batch
                author, paper, label = author.to(device), paper.to(device), label.to(device)
                out = model(author, paper)
                loss = criterion(out, label)
                train_tbar.set_postfix(loss="%.4f" % loss)
                writer.add_scalar('train/model_loss', loss, step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                if step % args.eval_every == 0:
                    model.eval()
                    correct, count = 0, 0
                    with torch.no_grad():
                        with tqdm(valid_loader, desc="Epoch %d" % epoch, ncols=100) as valid_tbar:
                            for batch in valid_tbar:
                                author, paper, label = batch
                                author, paper, label = author.to(device), paper.to(device), label.to(device)
                                out = model(author, paper)
                                pred = out.argmax(dim=1)
                                correct += (label == pred).float().sum()
                                count += label.shape[0]
                                valid_tbar.set_postfix(acc="%.4f" % (correct/count))
                            writer.add_scalar('eval/model_acc', correct/count, step)
                    model.train()
    writer.close()


if __name__ == "__main__":
    main()