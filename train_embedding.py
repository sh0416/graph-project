import os
import argparse
import linecache
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class FileDataset(Dataset):
    def __init__(self, fpath):
        self.fpath = fpath
        self.size = sum(1 for _ in open(fpath))

    def __getitem__(self, idx):
        return linecache.getline(self.fpath, idx+1).strip()
    
    def __len__(self):
        return self.size


class SkipGramNegativeSampleModel(nn.Module):
    def __init__(self, n_nodes, n_dims, window_size):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, n_dims)
        self.criterion = nn.BCEWithLogitsLoss()
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.neg_size = 4

    def forward(self, walk):
        anc = walk[:, :walk.shape[1]-self.window_size+1]  # [B, L-W]
        pos = torch.stack([walk[:, i:walk.shape[1]-self.window_size+1+i]
                           for i in range(1, self.window_size)], dim=2)  # [B, L-W, W]
        neg = torch.randint(low=1, high=self.n_nodes-1, size=(pos.shape[0], pos.shape[1], self.neg_size), device=pos.device) # [B, L-W, N]
        pos = torch.bmm(self.embed(pos.view(-1, self.window_size-1)), self.embed(anc.reshape(-1))[:, :, None])[:, :, 0] # [B*(L-W), W, D] x [B*(L-W), D, 1] = [B*(L-W), W, 1]
        neg = torch.bmm(self.embed(neg.view(-1, self.neg_size)), self.embed(anc.reshape(-1))[:, :, None])[:, :, 0]    # [B*(L-W), 4, D] x [B*(L-W), D, 1] = [B*(L-W), 4, 1]
        pos_label = torch.ones_like(pos)
        neg_label = torch.zeros_like(neg)
        return self.criterion(torch.cat([pos.view(-1), neg.view(-1)], dim=0),
                              torch.cat([pos_label.view(-1), neg_label.view(-1)], dim=0))


class TripletBCEEmbeddingModel(nn.Module):
    def __init__(self, n_nodes, n_dims):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, n_dims)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, anc, pos, neg):
        anc, pos, neg = self.embed(anc), self.embed(pos), self.embed(neg)
        pos = torch.bmm(anc[:, None, :], pos.transpose(1, 2))[:, 0, :]
        neg = torch.bmm(anc[:, None, :], neg.transpose(1, 2))[:, 0, :]
        pos_label = torch.ones_like(pos)
        neg_label = torch.zeros_like(neg)
        return self.criterion(torch.cat([pos, neg], dim=0),
                              torch.cat([pos_label, neg_label], dim=0))
        

class TripletEmbeddingModel(nn.Module):
    def __init__(self, n_nodes, n_dims):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, n_dims)
        self.criterion = nn.TripletMarginLoss()

    def forward(self, a, p, n):
        a, p, n = self.embed(a), self.embed(p), self.embed(n)
        loss = self.criterion(a, p[:, 0, :], p[:, 1, :])
        loss += self.criterion(a, p[:, 1, :], p[:, 2, :])
        loss += self.criterion(a, p[:, 2, :], n[:, 0, :])
        loss += self.criterion(a, n[:, 0, :], n[:, 1, :])
        loss += self.criterion(a, n[:, 1, :], n[:, 2, :])
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["node2vec", "ours"])
    parser.add_argument("--n_dims", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--save_every", type=int, default=4000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "node2vec":
        dataset = FileDataset(os.path.join("prep", "random_walks.csv"))
    else:
        dataset = FileDataset(os.path.join("prep", "triplet.csv"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.model == "node2vec":
        model = SkipGramNegativeSampleModel(12617, args.n_dims, 4)
    else:
        #model = TripletBCEEmbeddingModel(12617, args.n_dims) 
        model = TripletEmbeddingModel(12617, args.n_dims) 
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter()
    step = 0
    for epoch in range(args.epochs):
        with tqdm(loader, desc="Epoch %d" % epoch) as train_tbar:
            for batch in train_tbar:
                if args.model == "node2vec":
                    batch = torch.from_numpy(np.fromstring(','.join(batch), dtype=np.int64, sep=',')).view(args.batch_size, 101)
                else:
                    batch = torch.from_numpy(np.fromstring(','.join(batch).replace(' ', ','), dtype=np.int64, sep=',')).view(args.batch_size, 7)
                batch = batch.to(device)
                if args.model == "node2vec":
                    loss = model(batch)
                else:
                    loss = model(batch[:, 0], batch[:, 1:4], batch[:, 4:7])
                train_tbar.set_postfix(loss="%.4f" % loss)
                writer.add_scalar('train/loss', loss, step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                if step % args.save_every == 0:
                    torch.save(model.state_dict(), os.path.join("ckpt", "%s_%d_%06d.pt" % (args.model, args.n_dims, step)))
    writer.close()

if __name__ == "__main__":
    main()
