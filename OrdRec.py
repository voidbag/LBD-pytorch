import torch 
from torch import nn
import math

class BinsMode(nn.Module):
    def __init__(self,  bin_size=1., min_rating=1., max_rating=5.):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, inputs):
        output = (self.min_rating + torch.argmax(inputs, dim=-1).type(torch.float32) * self.bin_size)
        return output

class BinsMean(nn.Module):
    def __init__(self,  bin_size=1., min_rating=1., max_rating=5.):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, inputs):
        output = torch.sum(torch.mul(inputs, torch.arange(self.min_rating, self.max_rating + self.bin_size, self.bin_size, dtype=inputs.dtype, device=inputs.device)), dim=-1)
        return output

class GlobalBiasAdd(nn.Module):
    def __init__(self, global_bias):
        super().__init__()
        self.global_bias = global_bias

    def forward(self, inputs):
        return inputs + self.global_bias

def init_initializers(thresholds_use_items, t1_mode, b_mode, use_biases, bin_size, min_rating, max_rating=5.,override : dict = {}):
    initializers = {}
    for k in ["uid_t1", "iid_t1", "uid_beta", "iid_beta", "uid_bias", "iid_bias"]:
        initializers[k] = {"initializer":nn.init.constant_, "config": {"val": None}}

    if not thresholds_use_items:
        # Init t1 to 0.75
        initializers["uid_t1"]["config"]["val"] = min_rating + 0.5 * bin_size
        initializers["iid_t1"]["config"]["val"] = min_rating + 0.5 * bin_size
        # Init betas to log(bin_size)
        initializers["uid_beta"]["config"]["val"] = math.log(bin_size)
        initializers["iid_beta"]["config"]["val"] = math.log(bin_size)

        # Setting each item bias to the average ML-10m rating
        initializers["uid_bias"]["config"]["val"] = 0  # This will be ignored if `use_biases=False`
        initializers["iid_bias"]["config"]["val"] = 3.5

    else:
        # If t1_mode = 1:
        if t1_mode == 1:
            # uid_t1 = iid_t1 = 0.375
            initializers["uid_t1"]["config"]["val"] = 0.5 * (min_rating + 0.5 * bin_size)
            initializers["iid_t1"]["config"]["val"] = 0.5 * (min_rating + 0.5 * bin_size)
        # If t1_mode = 2:
        elif t1_mode == 2:
            # uid_t1 = iid_t1 = sqrt(0.75)
            initializers["uid_t1"]["config"]["val"] = math.sqrt(min_rating + 0.5 * bin_size)
            initializers["iid_t1"]["config"]["val"] = math.sqrt(min_rating + 0.5 * bin_size)
        # If b_mode = 1:
        if b_mode == 1:
            # b = log(bin_size/2)
            initializers["uid_beta"]["config"]["val"] = math.log(0.5 * bin_size)
            initializers["iid_beta"]["config"]["val"] = math.log(0.5 * bin_size)
        # If b_mode = 2:
        elif b_mode == 2:
            # b = 0.5 log(bin_size)
            initializers["uid_beta"]["config"]["val"] = 0.5 * math.log(bin_size)
            initializers["iid_beta"]["config"]["val"] = 0.5 * math.log(bin_size)

        # Setting each user and item bias to half of average ML-10m rating
        initializers["uid_bias"]["config"]["val"] = 1.75
        initializers["iid_bias"]["config"]["val"] = 1.75

    # Default embeddings
    initializers["uid_features"] = {"initializer": torch.nn.init.normal_, "config": {"std": 0.01}}
    initializers["iid_features"] = {"initializer": torch.nn.init.normal_, "config": {"std": 0.01}}

    # Add all
    for k, v in override.items():
        initializers[k] = v

    for k, v in initializers.items():
        initializers[k] = (v["initializer"], v["config"])

    return initializers

#def init_regularizers(override={}):
#
#    # Defaults
#    regularizers = {
#        "uid_features": keras.regularizers.L2(0.04),
#        "iid_features": keras.regularizers.L2(0.04),
#        "uid_bias": keras.regularizers.L2(0.),
#        "iid_bias": keras.regularizers.L2(0.),
#        "uid_beta": keras.regularizers.L2(0.001),
#        "iid_beta": keras.regularizers.L2(0.001)}
#
#    regularizers["uid_t1"] = regularizers["uid_bias"]
#    regularizers["iid_t1"] = regularizers["iid_bias"]
#
#    # Add all
#    for k, v in override.items():
#        regularizers[k] = v
#
#    return regularizers

def init_embedding(weight, init_tuple):
    initializer, kwargs = init_tuple
    initializer(weight, **kwargs)

class OrdRec(nn.Module):
    def __init__(self, num_users=0, num_items=0, num_hidden=512,
                 initializers = {},
                 regularizers = {},
                 thresholds_use_item = False,
                 t1_mode = 1,
                 beta_mode = 1,
                 use_biases = True,
                 bin_size = 0.5, min_rating = 0.5, max_rating = 5.):
        super().__init__()
        num_bins = int((max_rating - min_rating) / bin_size) + 1
        bins_params = {"bin_size": bin_size, "min_rating": min_rating, "max_rating": max_rating}

        self.num_bins = num_bins
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.thresholds_use_item = thresholds_use_item
        self.t1_mode = t1_mode
        self.beta_mode = beta_mode
        self.use_biases = use_biases
#        self.bins_mean = BinsMean(bin_size, min_rating, max_rating)
        self.bins_mean = BinsMean(**bins_params)
        self.bins_mode = BinsMode(**bins_params)
       
        initializers = init_initializers(thresholds_use_item, t1_mode, beta_mode, use_biases, **bins_params, override=initializers)
       
        # Features
        self.uid_features = nn.Embedding(num_users, num_hidden) # uid_features
        init_embedding(self.uid_features.weight, initializers["uid_features"])
        self.iid_features = nn.Embedding(num_items, num_hidden) # iid_features
        init_embedding(self.iid_features.weight, initializers["iid_features"])
 

        # Biases
        self.uid_bias = nn.Embedding(num_users, 1) # uid_bias
        init_embedding(self.uid_bias.weight, initializers["uid_bias"])
        self.iid_bias = nn.Embedding(num_items, 1) # iid_bias
        init_embedding(self.iid_bias.weight, initializers["iid_bias"])

        # T1's
        self.uid_t1 = nn.Embedding(num_users, 1) # uid_t1
        init_embedding(self.uid_t1.weight, initializers["uid_t1"])

        self.iid_t1 = nn.Embedding(num_items, 1) # iid_t1
        init_embedding(self.iid_t1.weight, initializers["iid_t1"])

        # Betas
        self.uid_beta = nn.Embedding(num_users, num_bins-2) # uid_beta
        init_embedding(self.uid_beta.weight, initializers["uid_beta"])


        self.iid_beta = nn.Embedding(num_items, num_bins-2) # iid_beta
        init_embedding(self.iid_beta.weight, initializers["iid_beta"])
   
    def forward(self, uid_input, iid_input):
        uid_features = self.uid_features(uid_input) # N x hidden_size
        iid_features = self.iid_features(iid_input) # N x hidden_size

        uid_bias = self.uid_bias(uid_input) # N x 1
        iid_bias = self.iid_bias(iid_input) # N x 1

        uid_t1 = self.uid_t1(uid_input) # N x 1
        iid_t1 = self.iid_t1(iid_input) # N x 1

        uid_beta = self.uid_beta(uid_input) # N x num_bins - 2
        iid_beta = self.uid_beta(iid_input) # N x num_bins - 2

        
        uTi = torch.linalg.vecdot(uid_features, iid_features, dim=1).unsqueeze(-1) # u·i N x 1

        # Determine model score
        if self.thresholds_use_item and (not self.use_biases):
            y_ui = uTi
        elif (not self.thresholds_use_item) and (not self.use_biases):
            y_ui = uTi + iid_bias # dot_plus_b_i
        else:
            y_ui = uTi + iid_bias + uid_bias # dot_plus_b_u_b_i
    
        # Determine t_1
        if not self.thresholds_use_item:
            t1 = uid_t1
        elif self.thresholds_use_item and self.t1_mode == 1:
            t1 = uid_t1 + iid_t1 # t1_sum
        elif self.thresholds_use_item and self.t1_mode == 2:
            t1 = torch.mul(uid_t1, iid_t1) # t1_prod
        # Determine gaps
        if not self.thresholds_use_item:
            beta = torch.exp(uid_beta) # beta
        elif self.thresholds_use_item and self.beta_mode == 1:
            beta = torch.exp(uid_beta) + torch.exp(iid_beta) # beta_sum
        elif self.thresholds_use_item and self.beta_mode == 2:
            beta = torch.exp(uid_beta + iid_beta) # beta_sum
    
        # Determine t_2 ... t_{N-1} and concat with t1
        beta_ext = torch.cat([torch.zeros_like(t1), beta], dim=-1) # beta_ext
        beta_cum = torch.cumsum(beta_ext, dim=-1) #beta_cum 
        T = t1 + beta_cum
        inf = torch.tensor([float('inf')] * T.size()[0], dtype=T.dtype, device=T.device).unsqueeze(-1)
        edges = torch.cat([T, inf], dim=-1)

        # Get CDF for t_1 ... t_{N-1}
        sigmoid_inputs = T - y_ui # sigmoid_inputs
        sigmoid = torch.sigmoid(sigmoid_inputs) # sigmoid
    
        # Add t_0 and t_N
        cdf = torch.cat([torch.zeros_like(t1), sigmoid, torch.ones_like(t1)], dim=-1) # cdf
    
        # Get bin scores
        bins_mass = cdf[:, 1:] - cdf[:, :-1] # bins_mass
    
        # Get point predictors
        bins_mean = self.bins_mean(bins_mass)
        bins_mode = self.bins_mode(bins_mass)
 
        # Gather outputs, including the binned distribution
        outputs = {"bins_mass": bins_mass, "bins_mean": bins_mean, "bins_mode": bins_mode, "edges": edges}
       
        return outputs



