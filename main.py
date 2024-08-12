import torch
from torch import nn
import pandas as pd
import json
from LBD import LBD
from OrdRec import OrdRec
from copy import deepcopy
from pathlib import Path
import glob
import argparse
import os
from loss import AdaptedCrossEntropy
from metric import RollingRMSE, RollingMAE, RollingAdaptedAccuracy, RollingAdpatedCrossEntropy
from collections import defaultdict
from collections import OrderedDict
from tqdm.auto import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dir-data", type=str, default="./data/k-folds/ml-10m/10-folds/")
parser.add_argument("--dir-output", type=str, default="./output")

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--cross-validation", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--model", type=str, default="LBD", choices=["LBD", "OrdRec"])
parser.add_argument("--model-config-json", type=str, default="./LBDA_512_sum_ab.json")

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--early-stopping-min-delta", type=float, default=0.0005)
parser.add_argument("--early-stopping-patience", type=int, default=10)

parser.add_argument("--bin_size", type=float, default=0.5)
parser.add_argument("--min_rating", type=float, default=0.5)
parser.add_argument("--max_rating", type=float, default=5.0)


class EarlyStopping():
    def __init__(self, min_delta=0.0005, patience=10, restore_best_state=True):
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_state = restore_best_state
        self.best_epoch = None
        self.best_state = None
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = None
        
    def get_best_state_n_epoch(self):
        if self.restore_best_state and self.best_state is not None:
            return (self.best_state, self.best_epoch)
            
    def __call__(self, model, loss, epoch):

        if self.restore_best_state and self.best_state is None:
            self.best_epoch = epoch
            self.best_state = model.state_dict()
            
        self.wait += 1
        
        if loss + self.min_delta < self.best_loss:
            print(f"{epoch} update best loss {self.best_loss} -> {loss}")
            self.best_loss = loss
            self.best_epoch = epoch
            if self.restore_best_state:
                self.best_state = deepcopy(model.state_dict())
            self.wait = 0
            return False
            
        if self.wait >= self.patience:
            print(f"{epoch} stop training {self.wait}")
            self.stopped_epoch = epoch
            return True
        print(f"wait {self.wait}")
        return False

class ResultCollector():
    def __init__(self, bin_size, min_rating, max_rating, li_keys = None):
        if li_keys is not None:
            self.li_keys = li_keys
        else:
            self.li_keys = ['alpha', 'beta', 'mu', 'upsilon', 'edges', 'bins_mass', 'mean', 'mode', 'bins_mode', 'bins_mean'] #LBD
            
        self.li_cols = ["uid", "iid", "gt"]
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.li_cols.extend(self.li_keys)
        self.li_x = [(i + 1) * bin_size for i in range(int((max_rating - min_rating) / bin_size + 1))]
        self.li_cols_mass = [ f"mass_{x :.1f}" for x in self.li_x]
        self.li_cols_edge = [ f"edge_{x :.1f}" for x in self.li_x]
        self.dict_raw = defaultdict(list)
        
    def _to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()
        
    def collect(self, uid, iid, gt, dict_out):
        self.dict_raw["uid"].append(self._to_numpy(uid))
        self.dict_raw["iid"].append(self._to_numpy(iid))
        self.dict_raw["gt"].append(self._to_numpy(gt))
        for key in self.li_keys:
            self.dict_raw[key].append(self._to_numpy(dict_out[key]))

    def to_df(self):
        set_excluded = {"edges", "bins_mass"}
        dict_df = OrderedDict()
        for col in self.li_cols:
            if col in set_excluded:
                continue
            value = np.concatenate(self.dict_raw[col]).flatten()
            dict_df[col] = value
        df = pd.DataFrame(dict_df)
        df[self.li_cols_mass] = np.concatenate(self.dict_raw["bins_mass"], axis=0)
        df[self.li_cols_edge] = np.concatenate(self.dict_raw["edges"], axis=0)
        return df

def init_model(path_config_json):
    with open(path_config_json, "r") as f:
        config = json.load(f)
    
    model = None
    model_name = config["name"] 
    model_params = config["model_params"]
    if model_name == "LBD":
        init_obj = model_params["initializer"]
        init_module = eval(init_obj["module"])
        params = init_obj["params"]
        model_params["initializer"] = (init_module, params)
        bias_initializers_obj = model_params["bias_initializers"]
        for idx in range(len(bias_initializers_obj)):
            module = eval(bias_initializers_obj[idx]["module"])
            params = bias_initializers_obj[idx]["params"]
            bias_initializers_obj[idx] = (module, params)
        model = LBD
    elif model_name == "OrdRec":
        init_obj = model_params["initializers"]
        for k, dict_param in init_obj.items():
            dict_param["initializer"] = eval(dict_param["initializer"])
        model = OrdRec
    else:
        assert False, f"LBD and OrdRec are only supported, not {model_name}" 
    
    return model(**model_params)

#["bins_mass", "bins_mean", "bins_mode", "edges"]
def evaluate(args, model, te_uid, te_iid, te_rating, collect=True):
    model.eval()
    batch_size = args.batch_size
    bin_size = args.bin_size
    min_rating = args.min_rating
    max_rating = args.max_rating

    num_iters =  (te_uid.size()[0] + (batch_size - 1)) // batch_size

    rolling_mode_accuracy = RollingAdaptedAccuracy(bin_size, min_rating, max_rating)
    rolling_mean_accuracy = RollingAdaptedAccuracy(bin_size, min_rating, max_rating)
    rolling_rmse = RollingRMSE(bin_size, min_rating, max_rating)
    
    if args.model == "LBD":
        out_features = ['alpha', 'beta', 'mu', 'upsilon', 'edges', 'bins_mass', 'mean', 'mode', 'bins_mode', 'bins_mean']
    else:
        assert args.model == "OrdRec"
        out_features = ["bins_mass", "bins_mean", "bins_mode", "edges"]

    collector = None
    if collect:
        collector = ResultCollector(bin_size, min_rating, max_rating, li_keys=out_features)
    for idx_iter in range(num_iters):
        idx_start = (idx_iter * batch_size)
        idx_end = min(idx_iter * batch_size + batch_size, te_uid.size()[0])
        _uid = te_uid[idx_start:idx_end]
        _iid = te_iid[idx_start:idx_end]
        _rating = te_rating[idx_start:idx_end]
        dict_out = model(_uid, _iid)
        mode = dict_out["bins_mode"]
        mean = dict_out["bins_mean"]
        rolling_rmse(_rating, mean)
        rolling_mode_accuracy(_rating, mode)
        rolling_mean_accuracy(_rating, mean)

        if collect:
            collector.collect(_uid, _iid, _rating, dict_out)
    
    rmse = rolling_rmse()
    mode_accuracy = rolling_mode_accuracy()
    mean_accuracy = rolling_mean_accuracy()
    if collect:
        df_result = collector.to_df() 
    else:
        df_result = None
    return rmse, mode_accuracy, mean_accuracy, df_result
    
def gen_input_tensors(df):
    tensor_uid = torch.tensor(df["uid"], dtype=torch.int32).cuda()
    tensor_iid = torch.tensor(df["iid"], dtype=torch.int32).cuda()
    tensor_rating = torch.tensor(df["rating"], dtype=torch.float32).cuda()
    return tensor_uid, tensor_iid, tensor_rating

def train(args, df_tr, df_te, epochs=None, use_stopper=True):
    if epochs is None:
        epochs = args.epochs
    else:
        assert use_stopper == False
    min_delta = args.early_stopping_min_delta
    patience = args.early_stopping_patience
    if use_stopper == False:
        min_delta = -10000000
    stopper = EarlyStopping(min_delta=min_delta, patience=patience)
    batch_size = args.batch_size

    tr_uid, tr_iid, tr_rating = gen_input_tensors(df_tr)
    te_uid, te_iid, te_rating = gen_input_tensors(df_te)

    model = init_model(args.model_config_json).cuda()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, eps=1e-07) # 1e-07 same as keras
    loss_fn = AdaptedCrossEntropy(bin_size=args.bin_size, min_rating=args.min_rating)

    for idx_epoch in tqdm(range(epochs)):
        new_idx =  torch.randperm(tr_uid.size()[0])
        new_tr_uid = tr_uid[new_idx]
        new_tr_iid = tr_iid[new_idx]
        new_tr_rating = tr_rating[new_idx]
        #torch.cuda.synchronize()
        num_iters =  (tr_uid.size()[0] + (batch_size - 1)) // batch_size
        model.train()

        is_last_epoch = (idx_epoch == epochs - 1)
        
        loss_per_batch = 0
        g_norm = 0
        for idx_iter in tqdm(range(num_iters), total=num_iters):
            optimizer.zero_grad()
            idx_start = (idx_iter * batch_size)
            idx_end = min(idx_iter * batch_size + batch_size, new_tr_uid.size()[0])
            _uid = new_tr_uid[idx_start:idx_end]
            _iid = new_tr_iid[idx_start:idx_end]
            _rating = new_tr_rating[idx_start:idx_end]

            dict_out = model(_uid, _iid)
            mass = dict_out["bins_mass"]
            loss = loss_fn(_rating, mass)
            
            #loss /= (idx_end - idx_start)
            loss.backward()
            optimizer.step()
            
            loss_per_batch += (loss.item() / (idx_end - idx_start) ) * batch_size
    
        loss_per_batch /= num_iters
    
        model.eval()
        rmse, mode_accuracy, mean_accuracy, _ = evaluate(args, model, te_uid, te_iid, te_rating, collect=False)
        
        print(f"epoch: {idx_epoch}")
        print(f"tr_loss_mean_over_batch: {loss_per_batch}")
        print(f"te_mode accuracy: {mode_accuracy}")
        print(f"te_mean accuracy: {mean_accuracy}")
        print(f"te_rmse loss    : {rmse}")
        for p in model.parameters():
            assert not p.isnan().any().item()
        
        stop = stopper(model, rmse, idx_epoch)
        if stop or is_last_epoch:
            state, best_epoch = stopper.get_best_state_n_epoch()
            model.load_state_dict(state)
            print(f"current_epoch: {idx_epoch}, best epoch: {stopper.best_epoch}, best rmse:{stopper.best_loss}")
            return model, best_epoch 

    assert False


def load_tr_te_from_dir(dir_data, tr_fname="tr.csv", te_fname="te.csv"):
    df_tr = pd.read_csv(os.path.join(dir_data, tr_fname))
    df_te = pd.read_csv(os.path.join(dir_data, te_fname))
    return df_tr, df_te

def main(args):

    li_fold = sorted([int(Path(path).parts[-2]) for path in glob.glob(f"{args.dir_data}/*/tr.csv")])
    assert (pd.Series(range(len(li_fold))) == pd.Series(li_fold)).all()

    dir_for_epoch = os.path.join(args.dir_data, "0")
    print("start tr_tr, tr_te learning for getting epoch")
    df_tr, df_te = load_tr_te_from_dir(dir_for_epoch, tr_fname="tr_tr.csv",  te_fname="tr_te.csv")
    model, best_epoch = train(args, df_tr, df_te)

    if args.cross_validation:
        max_fold = len(li_fold)
    else:
        max_fold = 1


    d = args.dir_output
    fname = os.path.splitext(os.path.basename(args.model_config_json))[0]
    d_out = os.path.join(d, fname)
    os.makedirs(d_out, exist_ok=True)
    with open(args.model_config_json, "r") as f:
        dict_config = json.load(f)
    dict_config["train_params"] = vars(args)
    with open(os.path.join(d_out, "config.json"), "w") as f:
        json.dump(dict_config, f, indent=2)

    for fold in range(max_fold):
        print(f"Start No.{fold} fold")
        dir_fold = os.path.join(args.dir_data, str(fold))
        df_tr, df_te = load_tr_te_from_dir(dir_for_epoch, tr_fname="tr.csv",  te_fname="te.csv")
        model, _best_epoch = train(args, df_tr, df_te, epochs=(best_epoch + 1), use_stopper=False)
        assert _best_epoch == best_epoch
        te_uid, te_iid, te_rating = gen_input_tensors(df_te)
        rmse, model_accuracy, mean_accuracy, df_result = evaluate(args, model, te_uid, te_iid, te_rating, collect=True)
        df_result["best_epoch"] = best_epoch
        df_result["fold"] = fold

        df_result.to_pickle(os.path.join(d_out, f"{fold}_df_eval.pkl"))
        torch.save(model.state_dict(), os.path.join(d_out, f"{fold}_model.pth"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

