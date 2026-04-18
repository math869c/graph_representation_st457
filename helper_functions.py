import numpy as np 
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

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

def create_data(x, batch_size=32, flatten_data = True, shuffle_train= True):
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
