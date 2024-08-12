import pandas as pd
import numpy as np
import glob
import os
import json

import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import argparse
from scipy import stats
# Set notebook mode to work in offline
#pyo.init_notebook_mode()
#pd.set_option("display.max_columns", None)

def gen_mae_var(df, bin_size, min_rating, max_rating):
    num_bins = int((max_rating - min_rating) / bin_size + 1)
    s_mae = (df["bins_mean"] - df["gt"]).abs()
    arr_x = np.expand_dims(np.array([(i + 1) * 0.5 for i in range(num_bins)]), axis=0)
    cols_mass = [f"mass_{(i + 1) * 0.5}" for i in range(num_bins)]
    s_var = (df[cols_mass] * np.power(arr_x, 2)).sum(axis=1) - df["bins_mean"].pow(2)
    return s_mae, s_var

def gen_cluster_to_plot(df, num_cluster=1000, rescale=True, quantile=True):
    df_mae_var = df[["bins_mae", "bins_var"]].sort_values(by="bins_var").reset_index(drop=True)
    sz_cluster = (df_mae_var.shape[0] + num_cluster) // num_cluster
    df_mae_var["cluster"] = (df_mae_var.index.to_series() / sz_cluster).astype(int)
    col_x = "predicted_variance"
    df_cluster = df_mae_var.groupby(by="cluster")[["bins_mae", "bins_var"]].mean().reset_index(names=[col_x])
    df_cluster[col_x] /= num_cluster
    return df_cluster

def gen_cluster_df(df, dict_config, reduce_mean=True, num_cluster=1000):
    df = df.copy()
    model_params = dict_config["model_params"]
    bin_size = model_params["bin_size"]
    min_rating = model_params["min_rating"]
    max_rating = model_params["max_rating"]
    mae, var = gen_mae_var(df, bin_size, min_rating, max_rating)
    df["bins_mae"] = mae
    df["bins_var"] = var

    li_fold = df["fold"].drop_duplicates().sort_values().tolist()
    _li_df = list()
    for fold in li_fold:
        _df = gen_cluster_to_plot(df.query(f"fold == {fold}"), num_cluster=1000)
        _df["fold"] = fold
        _li_df.append(_df)
    df_ret = pd.concat(_li_df, ignore_index=False)
    if reduce_mean:
        df_mean = df_ret.reset_index(drop=False).groupby(by="index")[["bins_mae", "bins_var"]].mean()
        df_std =  df_ret.reset_index(drop=False).groupby(by="index")[["bins_mae", "bins_var"]].std()
        df_mean["bins_mae_std"] = df_std["bins_mae"]
        df_mean["bins_var_std"] = df_std["bins_var"]
        df_ret = df_mean
        df_ret.sort_index(inplace=True)
        df_ret["predicted_variance"] = [(i + 1) / len(df_ret) for i in range(len(df_ret))]
    return df_ret

def gen_precision_one_to_plot(df, bin_size, min_rating, max_rating, thres=4.5):
    eps = 1e-7
    arr_x = np.arange(min_rating, max_rating + bin_size, bin_size)
    pred_high = arr_x[arr_x >= thres - eps]
    cols_high = [f"mass_{pred:.1f}" for pred in pred_high]
    col_pred = f"pred_+{thres:.1f}"
    col_gt = f"gt_+{thres:.1f}"
    df_to_plot = df[["uid", "iid"]].copy()
    df_to_plot[col_pred] = df[cols_high].sum(axis=1)
    df_to_plot[col_gt] = df["gt"] >= (thres - eps)

    df_to_plot = df_to_plot.sort_values(by=col_pred, ascending=False).reset_index(drop=True)
    df_to_plot = df_to_plot.loc[df_to_plot["uid"].drop_duplicates(keep="first").index].copy().reset_index(drop=True)
    col_precision = "precision@1"
    df_to_plot[col_precision] = df_to_plot[col_gt].cumsum() / np.arange(1, df_to_plot.shape[0] +1)
    col_x = "num_users"
    df_to_plot = df_to_plot.reset_index(names=[col_x])
    df_to_plot[col_x] += 1
    return df_to_plot

def gen_df_precision(df, dict_config, thres=4.5, reduce_mean=True):
    df = df.copy()
    model_params = dict_config["model_params"]
    bin_size = model_params["bin_size"]
    min_rating = model_params["min_rating"]
    max_rating = model_params["max_rating"]
    
    li_fold = df["fold"].drop_duplicates().sort_values().tolist()
    _li_df = list()
    for fold in li_fold:
        _df = gen_precision_one_to_plot(df.query(f"fold == {fold}"), bin_size, min_rating, max_rating, thres=thres)
        _df["fold"] = fold
        _li_df.append(_df)
    df_ret = pd.concat(_li_df, ignore_index=False)
    if reduce_mean:
        df_mean = df_ret.reset_index(drop=False).groupby(by="index")[["precision@1"]].mean()
        df_min =  df_ret.reset_index(drop=False).groupby(by="index")[["precision@1"]].min()
        df_max =  df_ret.reset_index(drop=False).groupby(by="index")[["precision@1"]].max()
        df_mean["precision@1_min"] = df_min["precision@1"]
        df_mean["precision@1_max"] = df_max["precision@1"]
        df_ret = df_mean
        df_ret.sort_index(inplace=True)
        df_ret.reset_index(names=["num_users"], inplace=True)
        df_ret["num_users"] += 1
    return df_ret

def update_precision_figure(fig, df, name, rgb, alpha=0.2, col_x="num_users", col_y="precision@1"):
    str_rgb = ",".join([str(num) for num in rgb])
    color = f"rgba({str_rgb},1)"
    color_shadow = f"rgba({str_rgb},{alpha})"

    fig.add_traces(go.Scatter(x=df[col_x], y=df[col_y],
                              mode="lines", line_color=color, name=name))

    fig.add_traces([go.Scatter(x=df[col_x], y = df[f"{col_y}_max"],
                               mode = 'lines', line_color = 'rgba(0,0,0,0)',
                               showlegend = False),
                    go.Scatter(x=df["num_users"], y = df[f"{col_y}_min"],
                               mode='lines', line_color='rgba(0,0,0,0)',
                               showlegend = False, fill='tonexty', fillcolor=color_shadow)])

parser = argparse.ArgumentParser()
parser.add_argument("--dir-lbd", type=str, default="./output/LBDA_512_sum_ab/")
parser.add_argument("--dir-ordrec", type=str, default="./output/OrdRec-UI_512/")
parser.add_argument("--out-prefix", type=str, default="./result-plot")

def main(args):
    dir_lbd = args.dir_lbd
    dir_ordrec = args.dir_ordrec
    pattern_fname = "*_df_eval.pkl"
    
    li_df_lbd = glob.glob(os.path.join(dir_lbd, pattern_fname))
    li_df_ordrec = glob.glob(os.path.join(dir_ordrec, pattern_fname))
    
    df_lbd = pd.concat([pd.read_pickle(path) for path in li_df_lbd], ignore_index=True).sort_values(by="fold", kind="stable").reset_index(drop=True)
    df_ordrec = pd.concat([pd.read_pickle(path) for path in li_df_ordrec], ignore_index=True).sort_values(by="fold", kind="stable").reset_index(drop=True)
    
    with open(os.path.join(dir_lbd, "config.json"), "r") as f:
        dict_config_lbd = json.load(f)
    with open(os.path.join(dir_ordrec, "config.json"), "r") as f:
        dict_config_ordrec = json.load(f)
    
    # Accuracy
    eps = 1e-7
    df_accuracy_lbd = df_lbd.groupby(by="fold")[["bins_mode", "gt"]].apply(lambda x: ((x["bins_mode"] - x["gt"]).abs() < eps).mean())
    df_accuracy_ordrec = df_ordrec.groupby(by="fold")[["bins_mode", "gt"]].apply(lambda x: ((x["bins_mode"] - x["gt"]).abs() < eps).mean())
    print(f"LBD-A     accuracy(std): {df_accuracy_lbd.mean():.5f}({df_accuracy_lbd.std():.6f})")
    print(f"OrdRec-UI accuracy(std): {df_accuracy_ordrec.mean():.5f}({df_accuracy_ordrec.std():.6f})")
    
    # Correlation
    mae, var = gen_mae_var(df_lbd, dict_config_lbd["model_params"]["bin_size"], dict_config_lbd["model_params"]["min_rating"], dict_config_lbd["model_params"]["max_rating"])
    df_lbd["bins_mae"] = mae
    df_lbd["bins_var"] = var
    
    mae, var = gen_mae_var(df_ordrec, dict_config_ordrec["model_params"]["bin_size"], dict_config_ordrec["model_params"]["min_rating"], dict_config_ordrec["model_params"]["max_rating"])
    df_ordrec["bins_mae"] = mae
    df_ordrec["bins_var"] = var
    
    df_pearson_lbd = df_lbd.groupby(by="fold")[["bins_mae", "bins_var"]].apply(lambda x: stats.pearsonr(x["bins_mae"], x["bins_var"]).statistic)
    df_pearson_ordrec = df_ordrec.groupby(by="fold")[["bins_mae", "bins_var"]].apply(lambda x: stats.pearsonr(x["bins_mae"], x["bins_var"]).statistic)
    
    df_kendall_lbd = df_lbd.groupby(by="fold")[["bins_mae", "bins_var"]].apply(lambda x: stats.kendalltau(x["bins_mae"], x["bins_var"]).statistic)
    df_kendall_ordrec = df_ordrec.groupby(by="fold")[["bins_mae", "bins_var"]].apply(lambda x: stats.kendalltau(x["bins_mae"], x["bins_var"]).statistic)
    
    li_index = ["Pearson's r (Linear Correlation)", "Kendall’s t (Rank   Correlation)"]
    values = [((f"{df_pearson_ordrec.mean():.5f}({df_pearson_ordrec.std():.5f})"), f"{df_pearson_lbd.mean():.5f}({df_pearson_lbd.std():.5f})", ),
              (f"{df_kendall_ordrec.mean():.5f}({df_kendall_ordrec.std():.5f})", f"{df_kendall_lbd.mean():.5f}({df_kendall_lbd.std():.5f})",)]
    li_cols = ["OrdRec-UI", "LBD-A"]
    df_correlation = pd.DataFrame(values, columns=li_cols, index=li_index)
    print(df_correlation)

	#Plot
    df_cluster_lbd = gen_cluster_df(df_lbd, dict_config_lbd)
    df_cluster_lbd["method"] = "LBD-A"
    df_cluster_ordrec = gen_cluster_df(df_ordrec, dict_config_ordrec)
    df_cluster_ordrec["method"] = "OrdRec-UI"
    df_to_plot = pd.concat([df_cluster_ordrec, df_cluster_lbd])
    fig = px.scatter(df_to_plot, x="predicted_variance", y="bins_mae", color="method")
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(width=1000, height=500)
    fig.write_html(f"{args.out_prefix}_variance_mae.html")

    fig = go.Figure()

    df_precision_lbd = gen_df_precision(df_lbd, dict_config_lbd)
    df_precision_lbd["method"] = "LBD-A"
    df_precision_ordrec = gen_df_precision(df_ordrec, dict_config_ordrec)
    df_precision_ordrec["method"] = "OrdRec-UI"

    update_precision_figure(fig, df_precision_ordrec, "OrdRec-UI", [0,0,255])
    update_precision_figure(fig, df_precision_lbd, "LBD-A", [255,0,0])

    fig.update_traces(line={'width': 1})
    fig.update_layout(width=1000, height=500)
    fig.update_layout(xaxis_title="# Users with a Recommendation (N)", yaxis_title="Precision@1")
    fig.update_yaxes(range=[0.6, 1.0])
    fig.update_xaxes(range=[np.log10(100), np.log10(55000)])
    fig.update_xaxes(type="log")
    fig.write_html(f"{args.out_prefix}_precision_1.html")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)



