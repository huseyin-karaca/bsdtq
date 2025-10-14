from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from SDT import SDT


def create_v_matrix(depth):
    num_internal = 2**depth - 1
    num_leaves = 2**depth
    V = -1*np.ones((num_internal, num_leaves), dtype=np.int32)
    #V = np.full((num_internal, num_leaves), np.nan, dtype=float)
    
    # Iterate level by level
    for m in range(depth):
        num_nodes_level = 2**m  # number of nodes at level m
        segment_length = 2**(depth - m)  # number of leaves each node at this level is responsible for
        half = segment_length // 2  # split point for left/right
        for k in range(num_nodes_level):
            # Compute the global index of the node in breadth-first order:
            node_index = 2**m - 1 + k
            
            # Determine the leaves corresponding to this node.
            seg_start = k * segment_length
            seg_end = seg_start + segment_length
            # For this internal node, assign left branch = 1 for the first half, right branch = 0 for the second half.
            V[node_index, seg_start:seg_start + half] = 1
            V[node_index, seg_start + half:seg_end] = 0
    return V



def compute_gradQ_batch_mse_bu(
    tree,
    Q: torch.Tensor,        # Q is a (37 x 37) matrix
    X: torch.Tensor,        # X is shape (batch_size, 37) of features
    y: torch.Tensor,        # y is shape (batch_size,) of targets
    boosting_iter: int
) -> torch.Tensor:
    """
    Compute the gradient of MSE loss wrt Q.
    MSE loss: (1/2)*sum_i ( f(Q, x_i) - y_i )^2, averaged over the batch.

    Args:
        tree: Your SDT model or structure with needed parameters in state_dict().
        Q:    The 37x37 matrix we differentiate wrt.
        X:    A mini-batch of inputs, shape (batch_size, 37).
        y:    Corresponding targets, shape (batch_size,).

    Returns:
        gradQ: shape (37, 37), the average gradient of the loss wrt Q.
    """

    # 1) Gather parameters from the tree
    W = tree.state_dict()["inner_nodes.0.weight"]           # shape (63, 38)
    gamma_ = tree.state_dict()["leaf_nodes.weight"][0, :]   # shape (64,)

    num_nodes = W.size(0)       # 63
    num_leaves = gamma_.size(0) # 64
    depth = int(np.log2(num_leaves))  # e.g. depth = 6

    # V: shape (63,64) or your actual structure (0/1).
    # If your code uses -1 for "not-on-path," adapt accordingly below.
    V = create_v_matrix(depth)   # or your real V

    batch_size = X.size(0)
    # input_dim = X.size(1)  # Should be 37
    output_dim, input_dim = Q.size()

    # 2) Initialize accumulator for gradients
    gradQ_batch = torch.zeros_like(Q)  # shape (37, 37)

    # preds = tree.forward(X@Q.transpose(0,1))

    # 3) Loop over each sample in the batch
    for i in tqdm(range(batch_size), desc = f"Gradient calculation for boosting step {boosting_iter}", total = batch_size,disable=True):
        #print(i)
        x_i = X[i]          # shape (37,)
        y_i = y[i]          # scalar

        # ---- Compute f(Q, x_i) ----
        # (a) z = Q @ x
        z = Q @ x_i  # shape (37,)

        # (b) data-augment: z1 has shape (38,) if W is 63x38
        z1 = torch.cat([torch.ones(1, device=z.device), z], dim=0)

        # (c) p[m] = sigmoid(W[m,:] . z1)
        p = torch.sigmoid(W @ z1)  # shape (63,)

        # (d) p_star[ell] = product over path -> 
        #     if V[m,ell]==1 => multiply p[m], if 0 => multiply (1 - p[m]).
        #     or skip if V[m,ell]==-1
        p_star = []
        for ell in range(num_leaves):
            p_star_l = 1.0
            for m in range(num_nodes):
                if V[m, ell] == 1:
                    p_star_l *= p[m]
                elif V[m, ell] == 0:
                    p_star_l *= (1 - p[m])
                # if -1 => skip
            p_star.append(p_star_l)

        # her adımda tek tek hesaplamaya gerek yok. tahmnileri fonksiyona input olarak verebiliriz. ya da içeride forward ile hesaplayabiliriz.
        # (e) f_i = sum_{ell} gamma_[ell] * p_star[ell]
        f_i = 0.0
        for ell in range(num_leaves):
            f_i += gamma_[ell] * p_star[ell]

        # (f) error_i = (f_i - y_i)
        #err = preds[i] - y_i
        err = f_i - y_i

        # ---- Now partial derivative wrt Q = err * d f_i / dQ ----
        gradQ_single = torch.zeros_like(Q)  # shape (37,37)

        # We'll do the triple loop from your code:
        for r in range(output_dim): 
            for c in range(input_dim):
                partial_rc = 0.0
                for ell in range(num_leaves):
                    sum_m = 0.0
                    for m in range(num_nodes):
                        # If your path logic requires skipping, do it:
                        if V[m, ell] != -1:
                            sum_m += (V[m, ell] - p[m]) * W[m, r+1]
                    # Multiply by gamma_[ell] * p_star[ell]
                    partial_rc += gamma_[ell] * p_star[ell] * sum_m
                # Multiply by x[c], as in your original formula
                partial_rc *= x_i[c]

                # Multiply by the error => derivative of MSE wrt Q
                partial_rc *= err.item()

                gradQ_single[r, c] = partial_rc

        gradQ_batch += gradQ_single

    # 4) Average over batch (typical mini-batch approach)
    gradQ_batch /= batch_size

    return gradQ_batch



def train_SDT(# Parameters
    # input_dim = 37,    # the number of input dimensions
    output_dim = 1,        # the number of outputs (i.e., # classes on MNIST)
    depth = 5,             # tree depth
    lamda = 1e-3,           # coefficient of the regularization term
    batch_size = 4096,
    lr = 1e-2,              # learning rate
    criterion = None,
    weight_decaly = 5e-4,   # weight decay
    #batch_size = 128,       # batch size
    epochs = 20, #50            # the number of training epochs
    log_interval = 400,     # the number of batches to wait before printing logs
    use_cuda = False,       # whether to use GPU
    eval = False,
    X = None,
    y = None,
    Q = None,
    boosting_iter = None,
    X_val = None,
    y_val = None):

    input_dim = X.shape[1]
    if Q is not None:
        X = X @ Q.transpose(0,1)
    dataset_train = TensorDataset(X,y)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    if eval:
        dataset_val = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    
    # Model and Optimizer
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)

    optimizer = torch.optim.Adam(tree.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decaly)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                        factor=0.5, patience=1)
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Utils
    best_validation_loss = float('inf')
    validation_loss_list = []
    training_loss_list = []
    # criterion = nn.CrossEntropyLoss()
    if criterion == "mse":
        criterion = nn.MSELoss()
    if criterion == "mae":
        criterion = nn.L1Loss()

    device = torch.device("cuda" if use_cuda else "cpu")

    pbar = tqdm(range(epochs), desc = f"SDT training for boosting step {boosting_iter}",total = epochs, dynamic_ncols=True, disable=True)
    for epoch in pbar:

        # Training
        tree.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):

            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device).view(-1,1)

            output, penalty = tree.forward(data, is_training_data=True)
            
            loss = criterion(output, target)
            loss += penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Print training status
            # if batch_idx % log_interval == 0:
            #     msg = (
            #         "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f}"
            #     )
            #     print(msg.format(epoch, batch_idx, loss))
            #     training_loss_list.append(loss.cpu().data.numpy())

        avg_train_loss = train_loss / len(train_loader)
        #print("Epoch: {:02d} | Average Training Loss: {:.5f}".format(epoch, avg_train_loss))
        #tqdm.write(f"Epoch {epoch}: Average Training Loss: {avg_train_loss:.5f}")
        #pbar.set_postfix(lossx=f"{avg_train_loss:.4f}")
        pbar.set_postfix({"lossx": f"{avg_train_loss:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.4f}"})
        

        # Evaluating
        if eval:
            tree.eval()
            val_loss = 0.0 

            for batch_idx, (data, target) in enumerate(val_loader):

                batch_size = data.size()[0]
                data, target = data.to(device), target.to(device).view(-1, 1)
                output = tree.forward(data)
                loss = criterion(output, target)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            validation_loss_list.append(avg_val_loss)

            if avg_val_loss < best_validation_loss:
                best_validation_loss = avg_val_loss

            print("Epoch: {:02d} | Average Validation Loss: {:.5f} | Historical Best: {:.5f}".format(epoch, avg_val_loss, best_validation_loss))

        # scheduler.step(avg_val_loss)
        scheduler.step()
        # pbar.set_postfix(LR=f"{scheduler.get_last_lr()[0]:.4f}")
        # print(f"Last LR: {scheduler.get_last_lr()[0]:.4f}")
        pbar.set_postfix({"lossx": f"{avg_train_loss:.4f}", "LR": f"{scheduler.get_last_lr()[0]:.4f}"})

    return tree



class BoostedSDTQ(BaseEstimator, RegressorMixin):
    def __init__(self,
                 output_dim=1,
                 depth=3,
                 lamda=1e-3,
                 iterate_until_converge = 100,
                 how_many_grad = 50,
                 criterion = "mse", #or "mae"
                 boosting_iters=3,
                 lr_q=1e-2,
                 batch_size=4096,
                 epochs=20,
                 lr_sdt=1e-2,
                 weight_decay=5e-4,
                 log_interval=400,
                 use_cuda=False,
                 eval=False):
        # hyperparams
        self.output_dim = output_dim
        self.how_many_grad = how_many_grad
        self.depth = depth
        self.lamda = lamda
        self.boosting_iters = boosting_iters
        self.lr_q = lr_q
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_sdt = lr_sdt
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.use_cuda = use_cuda
        self.eval = eval
        self.criterion = criterion
        self.models_ = {}
        self.Qs_ = {}
        self.iterate_until_converge = iterate_until_converge

    def fit(self, X, y):
        # scale targets
        # self._y_scaler = StandardScaler().fit(y.reshape(-1,1))
        # y_s = self._y_scaler.transform(y.reshape(-1,1)).ravel()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()
        y_s = y.copy()

        self.input_dim = X.shape[1]

        # initialize Q₀ = I
        Q = torch.eye(self.input_dim)
        Q = torch.ones(self.output_dim,self.input_dim)

        X_t = torch.tensor(X.values).float()
        y_t = torch.tensor(y_s).float()



        for it in range(self.boosting_iters):

            for _ in tqdm(range(self.iterate_until_converge),disable=False):
                # train one SDT on X @ Qᵀ
                tree = train_SDT(
                    output_dim=self.output_dim,
                    depth=self.depth,
                    lamda=self.lamda,
                    criterion=self.criterion,
                    batch_size=self.batch_size,
                    lr=self.lr_sdt,
                    weight_decaly=self.weight_decay,
                    epochs=self.epochs,
                    log_interval=self.log_interval,
                    use_cuda=self.use_cuda,
                    eval=self.eval,
                    X=X_t, y=y_t,
                    Q=Q,
                    boosting_iter=it
                )


                # randomly select indices to calculate gradients
                indices = torch.randperm(X_t.size(0))[:self.how_many_grad]

                # compute gradQ on the same batch
                gradQ = compute_gradQ_batch_mse_bu(
                    tree=tree, Q=Q, X=X_t[indices,:], y=y_t[indices], boosting_iter=it
                )
                #gradQ[gradQ > 0] = 0
                gradQ = F.normalize(gradQ, p =1)
                # simple gradient step
                Q = Q - self.lr_q * gradQ
                
                # Q = F.softmax(Q,dim = 1)
                Q = F.normalize(Q, p =1)
                #print("relevants:",Q[0,:5])
                #print("irrelevants:",Q[0,5:])

            self.models_[f"tree_biter_{it}"] = tree
            self.Qs_[f"Q_{it}"] = Q.clone()
        # store final Q also
        self.Qs_[f"Q_{self.boosting_iters}"] = Q.clone() 
        return self

    def predict(self, X):
        X_t = torch.tensor(X.values).float()
        # accumulate all trees’ outputs
        pred = torch.zeros(X_t.size(0), 1)
        for it in range(self.boosting_iters):
            Q = self.Qs_[f"Q_{it}"]
            tree = self.models_[f"tree_biter_{it}"]
            tree.eval()
            with torch.no_grad():
                out = tree.forward(X_t @ Q.t())
            pred += out
        # inverse‐scale and return numpy
        # pred = self._y_scaler.inverse_transform(pred.numpy())
        return pred.ravel()

    def score(self, X, y):
        # sklearn's score: higher is better, so we use –MSE
        y_pred = self.predict(X)
        return - mean_squared_error(y, y_pred)












class BoostedSDT(BaseEstimator, RegressorMixin):
    def __init__(self,
                 output_dim=1,
                 depth=3,
                 lamda=1e-3,
                 how_many_grad = 50,
                 criterion = "mse", #or "mae"
                 boosting_iters=3,
                 batch_size=4096,
                 epochs=20,
                 lr_sdt=1e-2,
                 weight_decay=5e-4,
                 log_interval=400,
                 use_cuda=False,
                 eval=False):
        # hyperparams
        self.output_dim = output_dim
        self.how_many_grad = how_many_grad
        self.depth = depth
        self.lamda = lamda
        self.boosting_iters = boosting_iters
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_sdt = lr_sdt
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.use_cuda = use_cuda
        self.eval = eval
        self.criterion = criterion

    def fit(self, X, y):
        # scale targets
        # self._y_scaler = StandardScaler().fit(y.reshape(-1,1))
        # y_s = self._y_scaler.transform(y.reshape(-1,1)).ravel()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()
        y_s = y.copy()

        self.input_dim = X.shape[1]

        X_t = torch.tensor(X.values).float()
        y_t = torch.tensor(y_s).float()

        self.models_ = {}

        for it in range(self.boosting_iters):
            # train one SDT on X @ Qᵀ
            tree = train_SDT(
                output_dim=self.output_dim,
                depth=self.depth,
                lamda=self.lamda,
                criterion=self.criterion,
                batch_size=self.batch_size,
                lr=self.lr_sdt,
                weight_decaly=self.weight_decay,
                epochs=self.epochs,
                log_interval=self.log_interval,
                use_cuda=self.use_cuda,
                eval=self.eval,
                X=X_t, y=y_t,
                boosting_iter=it
            )
            self.models_[f"tree_biter_{it}"] = tree

        return self

    def predict(self, X):
        X_t = torch.tensor(X.values).float()
        # accumulate all trees’ outputs
        pred = torch.zeros(X_t.size(0), 1)
        for it in range(self.boosting_iters):
            tree = self.models_[f"tree_biter_{it}"]
            tree.eval()
            with torch.no_grad():
                out = tree.forward(X_t)
            pred += out
        # inverse‐scale and return numpy
        # pred = self._y_scaler.inverse_transform(pred.numpy())
        return pred.ravel()

    def score(self, X, y):
        # sklearn's score: higher is better, so we use –MSE
        y_pred = self.predict(X)
        return - mean_squared_error(y, y_pred)



