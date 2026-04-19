import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import copy
from model_classes import *
import torch.optim as optim
# order of code 
# 1. create features and data to run models
# 2. training functions for LSTM
# 3. training functions for TGC
# 4. training functions for GAT
# 5. training functions for GAT+trans
# 6. metrics functions
# 7. Plotting functions for evalu

# 1. create features and data to run models
def moving_average(a, window):
    '''create moving average features, what they do in the paper'''
    return np.apply_along_axis(lambda x: np.convolve(x, np.ones(window)/window, mode='valid'), axis=0, arr=a)


def make_features(x, window = [5, 10, 20, 30]):
    '''Create features of moving averages'''
    x_returns = np.diff(np.log(x),axis=0)
    ma_dict = {w: moving_average(x_returns, w) for w in window}
    min_T = min(ma.shape[0] for ma in ma_dict.values())
    x_trim = x_returns[-min_T:]
    ma_trimmed = [ma_dict[w][-min_T:] for w in window]
    features = np.stack([x_trim] + ma_trimmed, axis=-1)
    return features 

def create_data(x, batch_size=32, flatten_data = True, flatten_time_features = False, shuffle_train= True):
    '''Prepare data for models'''
    features = make_features(x, window = [5, 10, 20, 30])
    # target all prices at time t
    target = features[:, :, 0]

    # split up the data
    n = len(target)
    train_end = n // 2
    val_end = train_end + n // 4
    train_raw = features[:train_end,:,:]
    val_raw   = features[train_end:val_end,:,:]
    test_raw  = features[val_end:,:, :]

    # Get original shapes for reshaping back
    train_shape = train_raw.shape
    val_shape = val_raw.shape
    test_shape = test_raw.shape

    # Reshape 3D data to 2D for StandardScaler (combine first two dimensions)
    train_reshaped = train_raw.reshape(-1, train_shape[-1])
    val_reshaped = val_raw.reshape(-1, val_shape[-1])
    test_reshaped = test_raw.reshape(-1, test_shape[-1])
    sc = StandardScaler()
    train_scaled_reshaped = sc.fit_transform(train_reshaped)
    val_scaled_reshaped   = sc.transform(val_reshaped)
    test_scaled_reshaped  = sc.transform(test_reshaped)

    # Reshape back to 3D after scaling
    train_scaled = train_scaled_reshaped.reshape(train_shape)
    val_scaled   = val_scaled_reshaped.reshape(val_shape)
    test_scaled  = test_scaled_reshaped.reshape(test_shape)

    # Concatenate along the time dimension (axis=0)
    full_scaled = np.concatenate((train_scaled, val_scaled, test_scaled), axis=0)

    seq_len = 8
    X, y = [], []
    for i in range(seq_len, len(full_scaled)):
        X.append(full_scaled[i-seq_len:i, :,:])
        y.append(full_scaled[i, :, 0])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    train_idx = train_end - seq_len
    val_idx   = val_end - seq_len

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val,   y_val   = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test,  y_test  = X[val_idx:], y[val_idx:]

    # keras can max input 3D and not 4D
    if flatten_data:
        (T, L, N, F) = X_train.shape
        X_train = X_train.reshape(T,L,N*F)
        (T, L, N, F) = X_val.shape
        X_val = X_val.reshape(T,L,N*F)
        (T, L, N, F) = X_test.shape
        X_test = X_test.reshape(T,L,N*F)
    elif flatten_time_features:
        (T, L, N, F) = X_train.shape
        X_train = X_train.reshape(T,N,L*F)
        (T, L, N, F) = X_val.shape
        X_val = X_val.reshape(T,N,L*F)
        (T, L, N, F) = X_test.shape
        X_test = X_test.reshape(T,N,L*F)

    # Convert to torch tensors and make into mini-batches (otherwise it can crash as it uses too much RAM)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32)

    print(X_train_t.shape)  # (T_train, 8, 460, 5)
    print(y_train_t.shape)  # (T_train, 460)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t, y_val_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train) # this includes all the data makes it into batches
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return X_train, y_train, X_val, y_val, X_test, y_test, sc, train_loader, val_loader, test_loader

# 2. training functions for LSTM and TGC
def directional_accuracy(preds, targets):
    return ((preds > 0) == (targets > 0)).float().mean().item()

def train_one_epoch_LSTM(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0

    for X_batch, y_batch in dataloader:

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        running_acc += directional_accuracy(preds.detach(), y_batch) * batch_size
        n += batch_size

    return running_loss / n, running_acc / n


def evaluate_LSTM(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            running_acc += directional_accuracy(preds, y_batch) * batch_size
            n += batch_size

    return running_loss / n, running_acc / n

def predict_final_model(model, dataloader):
    model.eval()
    preds = []

    with torch.no_grad():
        for X_batch, _ in dataloader:
            preds.append(model(X_batch).numpy())

    return np.concatenate(preds, axis=0)

def train_with_validation_LSTM(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10):
    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    history_metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_LSTM(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_LSTM(model, val_loader, criterion)

        history_metrics['train_loss'].append(train_loss)
        history_metrics['train_acc'].append(train_acc)
        history_metrics['val_loss'].append(val_loss)
        history_metrics['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history_metrics, best_val_loss



# 3. training functions for TGC
# functions for training, evaluating and predicting
def train_one_epoch(model, loader, A, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x_batch, y_batch in loader:
        optimizer.zero_grad()

        pred = model(x_batch, A)        # [B, N]
        loss = criterion(pred, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        total_count += x_batch.size(0)

    return total_loss / total_count

@torch.no_grad()
def evaluate(model, loader, A, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x_batch, y_batch in loader:
        pred = model(x_batch, A)
        loss = criterion(pred, y_batch)

        total_loss += loss.item() * x_batch.size(0)
        total_count += x_batch.size(0)

    return total_loss / total_count

@torch.no_grad()
def predict(model, loader, A):
    model.eval()
    preds = []
    ys = []

    for x_batch, y_batch in loader:
        pred = model(x_batch, A)

        preds.append(pred)
        ys.append(y_batch)

    preds = torch.cat(preds, dim=0)
    return preds

def train_with_validation(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10):
    best_val_loss = float("inf")
    best_state_dict = None
    patience_counter = 0

    history_metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history_metrics['train_loss'].append(train_loss)
        history_metrics['train_acc'].append(train_acc)
        history_metrics['val_loss'].append(val_loss)
        history_metrics['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history_metrics, best_val_loss

def training_loop_TGC(train_loader, val_loader, test_loader, y_test, adj_matrix, F, emb_dim, K_num_relations, num_epochs = 20):
    # do the training
    dict_adj_matrices = {'sector':    {'MSE':0, 'model':np.empty, 'matrix':adj_matrix[:,:,0].unsqueeze(-1), 'pred':np.empty},
                        'industry':  {'MSE':0, 'model':np.empty, 'matrix':adj_matrix[:,:,1].unsqueeze(-1), 'pred':np.empty},
                        'corre':     {'MSE':0, 'model':np.empty, 'matrix':adj_matrix[:,:,2].unsqueeze(-1), 'pred':np.empty},
                        'everything':{'MSE':0, 'model':np.empty, 'matrix':adj_matrix[:,:,:],               'pred':np.empty} }

    for key in dict_adj_matrices.keys():
        print(f'Doing model: {key}')
        A_loop = dict_adj_matrices[key]['matrix']
        K_num_relations = A_loop.shape[-1] # This will be 1 for single relations, and 3 for 'everything'

        model_TGC = TGCModel(num_features=F, emb_dim=emb_dim, gcn_dim=emb_dim, num_relations=K_num_relations)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_TGC.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
          train_loss = train_one_epoch(model_TGC, train_loader, A_loop, optimizer, criterion)
          val_loss = evaluate(model_TGC, val_loader, A_loop, criterion)

          print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        dict_adj_matrices[key]['model'] = model_TGC
        pred_test_TGC = predict(model_TGC, test_loader, A_loop)
        
        dict_adj_matrices[key]['pred'] = pred_test_TGC
        
        dict_adj_matrices[key]['MSE'] = np.mean((y_test-pred_test_TGC.numpy())**2,axis=0)

        metrics = compute_metrics(y_test, pred_test_TGC.numpy())
        dict_adj_matrices[key]['metrics'] = metrics
        
    return dict_adj_matrices

# 4. training functions for GAT
def train_one_epoch_GAT(model, loader, A_single_graph, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x_batch, y_batch in loader:
        optimizer.zero_grad()

        current_batch_size = x_batch.size(0)
        # Replicate the single graph adjacency matrix for the current batch
        A_batched = A_single_graph.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)

        pred = model(x_batch, A_batched)        # [B, N]
        loss = criterion(pred, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        total_count += x_batch.size(0)

    return total_loss / total_count

@torch.no_grad()
def evaluate_GAT(model, loader, A_single_graph, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x_batch, y_batch in loader:
        current_batch_size = x_batch.size(0)
        # Replicate the single graph adjacency matrix for the current batch
        A_batched = A_single_graph.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)

        pred = model(x_batch, A_batched)
        loss = criterion(pred, y_batch)

        total_loss += loss.item() * x_batch.size(0)
        total_count += x_batch.size(0)

    return total_loss / total_count

@torch.no_grad()
def predict_GAT(model, loader, A_single_graph):
    model.eval()
    preds = []
    ys = []

    for x_batch, y_batch in loader:
        current_batch_size = x_batch.size(0)
        # Replicate the single graph adjacency matrix for the current batch
        A_batched = A_single_graph.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)

        pred = model(x_batch, A_batched)

        preds.append(pred.cpu())
        ys.append(y_batch)

    preds = torch.cat(preds, dim=0)
    return preds


# 6. metrics functions
def compute_metrics(y_true, y_pred):
    # y_true: numpy array [T, N] - actual returns
    # y_pred: numpy array [T, N] - predicted returns

    # Convert to direction labels (1 = up, 0 = down)
    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)

    # Flatten for sklearn
    y_true_flat = y_true_dir.flatten()
    y_pred_flat = y_pred_dir.flatten()

    # Classification metrics
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)

    # Trading strategy: buy if predicted up, short if predicted down
    trade_sign = np.where(y_pred_dir == 1, 1, -1)
    strategy_returns = y_true * trade_sign
    daily_returns = strategy_returns.mean(axis=1)

    # Financial metrics
    return_ratio = daily_returns.sum()
    std = daily_returns.std()
    sharpe = (daily_returns.mean() / std) * np.sqrt(252) if std > 0 else 0.0

    # MSE 
    MSE = np.mean((y_true - y_pred) ** 2)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'mcc': mcc, 
        'return_ratio': return_ratio, 
        'sharpe': sharpe,
        'MSE': MSE
    }

# 7. Plotting functions for evalu
def plot_results(y_test, pred, open_prices_interp, nr_firms=10):
    seq_len = 8
    max_left_from_ma = int(30 // 2 - 1)
    start_date_for_plot = 1 + max_left_from_ma + seq_len
    tickers = open_prices_interp.columns
    dates = open_prices_interp.index[start_date_for_plot:start_date_for_plot + len(y_test)]
    for i in range(nr_firms):
        plt.plot(dates, y_test[:, i], color='red', label='Real Stock Price')
        plt.plot(dates, pred[:, i], color='blue', label='Predicted Stock Price')
        plt.xlabel('Time')
        plt.ylabel('S&P 500 Open Price')
        plt.title(f"Firm: {tickers[i]}")
        plt.legend()
        plt.show()

def print_box_plots(MSE_dict):
    plt.figure(figsize=(12,5))

    val_list = []
    label_list = []
    for key, val in MSE_dict.items():
        val_list.append(val)
        label_list.append(key)

    plt.boxplot(val_list, labels=label_list)

    plt.title("MSE comparison across models")
    plt.ylabel("MSE")
    plt.show()