import torch
from torch import nn

class UpsilonNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.tensor(1e-6)

    def forward(self, u, i):
        return torch.maximum(torch.mul(torch.norm(u, dim=1, keepdim=True), torch.norm(i, dim=1, keepdim=True)), self.eps)


class UpsilonDot(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.tensor(1e-6)

    def forward(self, u, i):
        return torch.maximum(torch.abs(torch.sum(torch.mul(u, i), dim=-1, keepdim=True)), self.eps)


class UpsilonSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.tensor(1e-6)

    def forward(self, u, i):
        return torch.maximum(torch.norm(u + i, dim=1, keepdim=True), self.eps)


class BiasMuUpsilon(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        lower_mu, upper_mu = torch.sqrt(0.5) - 0.1, torch.sqrt(0.5) + 0.1
        self.uid_mu_emb = nn.Embedding(num_users, 1)
        self.iid_mu_emb = nn.Embedding(num_items, 1)
        nn.init.uniform_(self.uid_mu_emb.weight, lower_mu, upper_mu)
        nn.init.uniform_(self.iid_mu_emb.weight, lower_mu, upper_mu)

        lower_upsilon, upper_upsilon = 1 - 0.1, 1 + 0.1 
        self.uid_upsilon_emb = nn.Embedding(num_users, 1)
        self.iid_upsilon_emb = nn.Embedding(num_items, 1)
        nn.init.uniform_(self.uid_upsilon_emb.weight, lower_upsilon, upper_upsilon)
        nn.init.uniform_(self.iid_upsilon_emb.weight, lower_upsilon, upper_upsilon)
        self.eps = torch.tensor(1e-6)
        self.eps_param = torch.tensor(1e-2)

    def forward(self, uid, iid, mu, upsilon): # returns mu, upsilon, alpha, beta
        uid_mu_bias = self.uid_mu_emb(uid)
        iid_mu_bias = self.iid_mu_emb(iid)
        uid_upsilon_bias = self.uid_upsilon_emb(uid)
        iid_upsilon_bias = self.iid_upsilon_emb(iid)
       
        upsilon = torch.clamp(torch.multiply(upsilon, torch.multiply(uid_upsilon_bias, iid_upsilon_bias)), 1e-6, 15.0)

        prod_mu_bias =  torch.multiply(uid_mu_bias, iid_mu_bias)
        mu = torch.where(mu < prod_mu_bias,
                         0.5 * mu / torch.maximum(prod_mu_bias, self.eps),
                         0.5 + 0.5 * (mu - prod_mu_bias) / torch.maximum(1 - prod_mu_bias, self.eps)) 

        alpha = torch.maximum(mu * upsilon, self.eps_param)
        beta = torch.maximum(upsilon - alpha, self.eps_param) # Equivalent to (1 - mu ) * upsilon
        
        return mu, upsilon, alpha, beta


# Uid/iid learned alpha/beta adjustments
class BiasAlphaBeta(nn.Module):
    def __init__(self, num_users, num_items, initializers):
        super().__init__()
        # for compatibility in original repository
        assert len(initializers) == 2
        alpha_init_fn = initializers[0][0]
        alpha_init_kwargs = initializers[0][1]
        self.uid_alpha_emb = nn.Embedding(num_users, 1)
        self.iid_alpha_emb = nn.Embedding(num_items, 1)

        alpha_init_fn(self.uid_alpha_emb.weight, **alpha_init_kwargs)
        alpha_init_fn(self.iid_alpha_emb.weight, **alpha_init_kwargs)
        beta_init_fn = initializers[1][0]
        beta_init_kwargs = initializers[1][1]
        self.uid_beta_emb = nn.Embedding(num_users, 1)
        self.iid_beta_emb = nn.Embedding(num_items, 1)
        beta_init_fn(self.uid_beta_emb.weight, **beta_init_kwargs)
        beta_init_fn(self.iid_beta_emb.weight, **beta_init_kwargs)

        self.g_alpha_bias = GlobalBiasAdd(0.3)
        self.g_beta_bias = GlobalBiasAdd(0.3)
        self.eps_param = torch.tensor(1e-2)

    def forward(self, uid, iid, mu, upsilon):
        uid_alpha_emb = self.uid_alpha_emb(uid)
        iid_alpha_emb = self.iid_alpha_emb(iid)
        uid_beta_emb = self.uid_beta_emb(uid)
        iid_beta_emb = self.iid_beta_emb(iid)

        alpha = torch.maximum(mu * upsilon, self.eps_param)
        beta = torch.maximum(upsilon - alpha, self.eps_param) # Equivalent to (1 - mu ) * upsilon
        alpha = self.g_alpha_bias(alpha)
        beta = self.g_beta_bias(beta)
        alpha = torch.maximum(alpha + uid_alpha_emb + iid_alpha_emb, self.eps_param)
        beta = torch.maximum(beta + uid_beta_emb + iid_beta_emb, self.eps_param)
        return mu, upsilon, alpha, beta


class LBD(nn.Module):
    def __init__(self, num_users=0, num_items=0,
        num_hidden=512, upsilon_layer_id=3,
        bin_size=1., min_rating=1., max_rating=5.,
        bias_mode=1,
        initializer = (torch.nn.init.normal_, {}),
        bias_initializers=[(torch.nn.init.ones_, {})]*2,
        regularize_activity=True,
        split_embeddings=False, adaptive_edges=False,
        ):
        super().__init__()

        self.adaptive_edges = adaptive_edges
        # Get each input to the model
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.uid_emb = nn.Embedding(num_users, num_hidden)
        self.iid_emb = nn.Embedding(num_items, num_hidden)
        if initializer is not None:
            initializer, params = initializer[0], initializer[1]
            initializer(self.uid_emb.weight, **params)
            initializer(self.iid_emb.weight, **params)

        self.split_embeddings = split_embeddings
        self.uid_confidence_emb = nn.Embedding(num_users, num_hidden) if split_embeddings else self.uid_emb
        self.iid_confidence_emb = nn.Embedding(num_items, num_hidden) if split_embeddings else self.iid_emb

        if upsilon_layer_id == 1:
            self.upsilon_layer =  UpsilonNorm()
        elif upsilon_layer_id == 2:
            self.upsilon_layer = UpsilonDot()
        elif upsilon_layer_id == 3:
            self.upsilon_layer = UpsilonSum()
        else:
            assert False, f"upsilon_layer_id should belong to {1, 2, 3}, not {upsilon_layer_id}!"

        if bias_mode == 1:
            self.bias_layer = BiasAlphaBeta(num_users, num_items, bias_initializers)
        elif bias_mode == 2:
            self.bias_layer = BiasMuUpsilon(num_users, num_items)
        else:
            assert False, f"biad_mode should belong to {1, 2}, not {bias_mode}!"

        if adaptive_edges:
            self.bin_layer = BetaBinsMassAdaptive(num_users, num_items, bin_size, min_rating, max_rating)
        else:
            self.bin_layer = BetaBinsMass(bin_size, min_rating, max_rating)

        self.eps = torch.tensor(1e-6)

    def forward(self, uid, iid):
        uid_features = self.uid_emb(uid)#[:, :-1]
        iid_features = self.iid_emb(iid)#[:, :-1]
        uid_confidence_features = self.uid_confidence_emb(uid)#[:, :-1]
        iid_confidence_features = self.iid_confidence_emb(iid)#[:, :-1]
        # Forward steps
        dot = torch.linalg.vecdot(uid_features, iid_features).unsqueeze(-1) # u·i
        uid_norm = torch.norm(uid_features, dim=1, keepdim=True)# ||u||
        iid_norm = torch.norm(iid_features, dim=1, keepdim=True)# ||i||
        len_prod = torch.mul(uid_norm, iid_norm) # ||u||·||i||
        mu = torch.clamp(0.5 + 0.5 * dot / torch.maximum(len_prod, self.eps), 1e-6, 1 - 1e-6)
        upsilon = self.upsilon_layer(uid_confidence_features, iid_confidence_features)
        mu, upsilon, alpha, beta = self.bias_layer(uid, iid, mu, upsilon) 

        outputs = {"alpha": alpha, "beta": beta, "mu": mu, "upsilon": upsilon}

        beta_bins_mass, edges = self.bin_layer(uid, iid, alpha, beta)


        outputs["edges"] = edges

        metric_params = {"bin_size": self.bin_size, "min_rating": self.min_rating, "max_rating": self.max_rating}

        beta_bins_mean = BetaBinsMean(self.bin_size, self.min_rating, self.max_rating, name="beta_bins_mean")(beta_bins_mass)

        beta_bins_mode = BetaBinsMode(self.bin_size, self.min_rating, self.max_rating, name="beta_bins_mode")(beta_bins_mass)

        beta_mean = BetaMean(self.bin_size, self.min_rating, self.max_rating,  name="beta_mean")([alpha, beta])
        #beta_median = BetaMedian(**metric_params, name="beta_median")([alpha, beta])
        beta_mode = BetaMode(self.bin_size, self.min_rating, self.max_rating, name="beta_mode")([alpha, beta])

        outputs.update({"bins_mass": beta_bins_mass, "mean": beta_mean, "mode": beta_mode,# "median": beta_median,
                        "bins_mode": beta_bins_mode, "bins_mean": beta_bins_mean})

        
        return outputs


class GlobalBiasAdd(nn.Module):
    def __init__(self, bias=1., **kwargs):
        super().__init__()
        self.global_bias = nn.Parameter(torch.tensor(bias), requires_grad=True)

    def forward(self, inputs):
        return inputs + self.global_bias


class BetaBinsMode(nn.Module):
    def __init__(self,  bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, inputs):
        output = self.min_rating + torch.argmax(inputs, dim=-1).type(torch.float32) * self.bin_size
        return output


class BetaBinsMean(nn.Module):
    def __init__(self,  bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, inputs):
        output = torch.sum(torch.mul(inputs, torch.arange(self.min_rating, self.max_rating + self.bin_size, self.bin_size, dtype=inputs.dtype, device=inputs.device)), dim=-1)  # Will not work properly with bin_size!=1.
        return output


class BetaBinsMass(nn.Module):
    def __init__(self, bin_size=1., min_rating=1, max_rating=5, adaptive_edges=False):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num_bins = (self.max_rating - self.min_rating) / self.bin_size + 1
        self.bins = torch.arange(self.min_rating, self.max_rating + bin_size, bin_size)
        self.bins_01 = torch.arange(1, self.num_bins + bin_size) / self.num_bins

    @staticmethod
    def cdf(x, a, b):
        return torch.special.betainc(x, a, b)
        # return 1 - (1 - x ** a) ** b

    def forward(self, uid, iid, alpha, beta):
        # Calculate cdf at each bin end: (None, num_bins)
        cdf = self.cdf(self.bins_01[:-1], alpha, beta)
        # Replace last cdf bin to avoid issues with calculating d/dx of cdf with convex tails
        cdf = torch.cat([cdf, torch.ones_like(alpha)], dim=-1)
        # Calculate mass in each bin: (None, num_bins)
        mass = torch.cat([cdf[:, :1], torch.diff(cdf)], dim=-1)
        # Output tensors: prediction, mass
        return mass, self.bins_01.repeat(uid.size()[0], 1)


class BetaBinsMassAdaptive(nn.Module):
    def __init__(self, num_users, num_items, bin_size=1., min_rating=1, max_rating=5):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num_bins = int((self.max_rating - self.min_rating) / self.bin_size + 1)
        self.uid_bin_emb = nn.Embedding(num_users, self.num_bins)
        self.iid_bin_emb = nn.Embedding(num_items, self.num_bins)
        nn.init.ones_(self.uid_bin_emb.weight)
        nn.init.ones_(self.iid_bin_emb.weight)

    @staticmethod
    def cdf(x, a, b):
        return torch.special.betainc(x, a, b)

    def forward(self, uid, iid, alpha, beta):
        uid_bin_size_terms = self.uid_bin_emb(uid)
        iid_bin_size_terms = self.iid_bin_emb(iid)
        
        ui_bin_size_terms = torch.exp(uid_bin_size_terms + iid_bin_size_terms)
        ui_bin_size_terms_norm = ui_bin_size_terms / torch.sum(ui_bin_size_terms, dim=-1, keepdim=True)
        edges = torch.cumsum(ui_bin_size_terms_norm, dim=-1)
        bins_01 = edges
        alpha = alpha.repeat(1, edges.size()[-1] - 1)
        beta = beta.repeat(1, edges.size()[-1] - 1)
        # Calculate cdf at each bin end: (None, num_bins)
        cdf = self.cdf(bins_01[:, :-1], alpha, beta)
        # Replace last cdf bin to avoid issues with calculating d/dx of cdf with convex tails
        cdf = torch.cat([cdf, torch.ones_like(alpha[:,:1])], dim=-1)
        # Calculate mass in each bin: (None, num_bins)
        mass = torch.cat([cdf[:, :1], torch.diff(cdf)], dim=-1)
        # Output tensors: prediction, mass
        return mass, edges

class BetaMean(nn.Module):
    def __init__(self, bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, inputs):
        alpha, beta = inputs
        output = (alpha / (alpha + beta))[:, 0] * self.max_rating
        return torch.clamp(output + self.bin_size/2., self.min_rating, self.max_rating)  # (None,)

class BetaMode(nn.Module):
    def __init__(self, bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__()
        self.bin_size = bin_size 
        self.min_rating = min_rating 
        self.max_rating = max_rating 

    def mode(self, alpha, beta):
        a_above_1, b_above_1 = alpha > 1, beta > 1
        a_b = a_above_1 & b_above_1
        a_not_b = a_above_1 & ~b_above_1
        not_a_b = ~a_above_1 & b_above_1
        a_above_b = alpha > beta
        b_above_a = alpha < beta
        return torch.where(
            a_b,
            self._default_mode(alpha, beta),
            torch.where(
                a_not_b,
                self.max_rating,
                torch.where(
                    not_a_b,
                    0.,
                    torch.where(
                        a_above_b,
                        self.max_rating,
                        torch.where(b_above_a,
                                 0.,
                                 0.5*self.max_rating)
                    )
                )
            )
        )

    def _default_mode(self, alpha, beta):
        return ((alpha - 1) / (alpha + beta - 2)) * self.max_rating

    def forward(self, inputs):
        alpha, beta = inputs[0][:,0], inputs[1][:,0]
        output = self.mode(alpha, beta)
        return torch.clamp(output + self.bin_size/2., self.min_rating, self.max_rating)  # (None,)

