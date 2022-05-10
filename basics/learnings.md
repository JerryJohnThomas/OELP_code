# learnings


## site0
https://analyticsindiamag.com/hands-on-guide-to-pytorch-geometric-with-python-code/


## medium1
[link](https://colab.research.google.com/github/arangodb/interactive_tutorials/blob/master/notebooks/Integrate_ArangoDB_with_PyG.ipynb#scrollTo=gkOJMvuuq59A)

```python

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # z_dict contains dictionary of movie and user embeddings returned from GraphSage
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)



# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:
with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['user', 'movie'].edge_label_index)
    target = train_data['user', 'movie'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'movie'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)



for epoch in range(1, 300):
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')

```



## medium 2 


[link](https://medium.com/@khang.pham.exxact/gnn-demo-using-pytorch-lightning-and-pytorch-geometric-40884cb9bbbd)


```python

class PlainGCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PlainGCN, self).__init__()
        self.num_features = kwargs["num_features"] \
            if "num_features" in kwargs.keys() else 3
        self.num_classes = kwargs["num_classes"] \
            if "num_classes" in kwargs.keys() else 2
        # hidden layer node features
        self.hidden = 256
        self.model = Sequential("x, edge_index, batch_index", [\                
                (GCNConv(self.num_features, self.hidden), \
                    "x, edge_index -> x1"),
                (ReLU(), "x1 -> x1a"),\                                         
                (Dropout(p=0.5), "x1a -> x1d"),\                                
                (GCNConv(self.hidden, self.hidden), "x1d, edge_index -> x2"), \ 
                (ReLU(), "x2 -> x2a"),\                                         
                (Dropout(p=0.5), "x2a -> x2d"),\                                
                (GCNConv(self.hidden, self.hidden), "x2d, edge_index -> x3"), \ 
                (ReLU(), "x3 -> x3a"),\                                         
                (Dropout(p=0.5), "x3a -> x3d"),\                                
                (GCNConv(self.hidden, self.hidden), "x3d, edge_index -> x4"), \ 
                (ReLU(), "x4 -> x4a"),\                                         
                (Dropout(p=0.5), "x4a -> x4d"),\                                
                (GCNConv(self.hidden, self.hidden), "x4d, edge_index -> x5"), \ 
                (ReLU(), "x5 -> x5a"),\                                         
                (Dropout(p=0.5), "x5a -> x5d"),\                                
                (global_mean_pool, "x5d, batch_index -> x6"),\                  
                (Linear(self.hidden, self.num_classes), "x6 -> x_out")])    
       
    def forward(self, graph_data):
        x, edge_index, batch = graph_data.x, graph_data.edge_index,\
                    graph_data.batch
        x_out = self.model(x, edge_index, batch)
        return x_out
```




Eval func
```python

def evaluate(model, test_loader, save_results=True, tag="_default", verbose=False):
    # get test accuracy score
    num_correct = 0.
    num_total = 0.
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0
    total_batches = 0
    for batch in test_loader:
        pred = model(batch.to(my_device))
        loss = criterion(pred, batch.y.to(my_device))
        num_correct += (pred.argmax(dim=1) == batch.y).sum()
        num_total += pred.shape[0]
        total_loss += loss.detach()
        total_batches += batch.batch.max()
    test_loss = total_loss / total_batches
    test_accuracy = num_correct / num_total
    if verbose:
        print(f"accuracy = {test_accuracy:.4f}")
    results = {"accuracy": test_accuracy, \
        "loss": test_loss, \
        "tag": tag }
    return results
```


The Training Loop
```
def train_model(model, train_loader, criterion, optimizer, num_epochs=1000, \
        verbose=True, val_loader=None, save_tag="default_run_"):
    ## call validation function and print progress at each epoch end
    display_every = 1 #num_epochs // 10
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(my_device)
    # we'll log progress to tensorboard
    log_dir = f"lightning_logs/plain_model_{str(int(time.time()))[-8:]}/"
    writer = SummaryWriter(log_dir=log_dir)
    t0 = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch.to(my_device))
        loss = criterion(pred, batch.y.to(my_device))
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
        batch_count += 1
    mean_loss = total_loss / batch_count
    writer.add_scalar("loss/train", mean_loss, epoch)
    if epoch % display_every == 0:
        train_results = evaluate(model, train_loader, \
        tag=f"train_ckpt_{epoch}_", verbose=False)
        train_loss = train_results["loss"]
        train_accuracy = train_results["accuracy"]
    if verbose:
        print(f"training loss & accuracy at epoch {epoch} = "\
        f"{train_loss:.4f} & {train_accuracy:.4f}")
    if val_loader is not None:
        val_results = evaluate(model, val_loader, \
        tag=f"val_ckpt_{epoch}_", verbose=False)
        val_loss = val_results["loss"]
        val_accuracy = val_results["accuracy"]
    if verbose:
        print(f"val. loss & accuracy at epoch {epoch} = "\
        f"{val_loss:.4f} & {val_accuracy:.4f}")
        else:
        val_loss = float("Inf")
        val_acc = - float("Inf")
    writer.add_scalar("loss/train_eval", train_loss, epoch)
    writer.add_scalar("loss/val", val_loss, epoch)
    writer.add_scalar("accuracy/train", train_accuracy, epoch)
    writer.add_scalar("accuracy/val", val_accuracy, epoch)


```

## set3 medium3



[link](https://medium.com/stanford-cs224w/predicting-subject-areas-of-cs-papers-using-graph-neural-networks-via-pyg-ea107fd4f571)


```python
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        ### initialize a list of GCNconvs
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
 
        ### initialize a list of Batch-Norms
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
      
        ### add as much GCNConvs and Batch-Norms as there are layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
 
        ### the dropout rate
        self.dropout = dropout
 
    def forward(self, x, adj_t):
        # x is the tensor containing node features
        # adj_t is the adjacency list (also tensor)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
```


```python
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    # the difference between this function and the one used for the mlp is
    # that we give the model all nodes as input and then select the predictions
    # on the train set for the calculation of the loss
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()
```


## src4

set of notebooks here
https://github.com/mnslarcher/cs224w-slides-to-code/tree/main/notebooks











