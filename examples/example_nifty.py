import sys
sys.path.append('..')
from pygdebias.debiasing import NIFTY
from pygdebias.datasets import Bail

import numpy as np
from collections import defaultdict
import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


setup_seed(11)

bail = Bail()
adj, feats, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
    bail.adj(),
    bail.features(),
    bail.idx_train(),
    bail.idx_val(),
    bail.idx_test(),
    bail.labels(),
    bail.sens(),
    bail.sens_idx()
)

# Initiate the model (with searched parameters).
model = NIFTY(
    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx, num_hidden=128, num_proj_hidden=128, lr=0.001, weight_decay=1e-5, drop_edge_rate_1=0.1, drop_edge_rate_2=0.1, drop_feature_rate_1=0.1, drop_feature_rate_2=0.1, encoder="gcn", sim_coeff=0.5, nclass=1, device="cuda").cuda()
model.fit()


# Evaluate the model.

(
    ACC,
    AUCROC,
    F1,
    ACC_sens0,
    AUCROC_sens0,
    F1_sens0,
    ACC_sens1,
    AUCROC_sens1,
    F1_sens1,
    SP,
    EO,
) = model.predict()

print("ACC:", ACC)
print("AUCROC: ", AUCROC)
print("F1: ", F1)
print("ACC_sens0:", ACC_sens0)
print("AUCROC_sens0: ", AUCROC_sens0)
print("F1_sens0: ", F1_sens0)
print("ACC_sens1: ", ACC_sens1)
print("AUCROC_sens1: ", AUCROC_sens1)
print("F1_sens1: ", F1_sens1)
print("SP: ", SP)
print("EO:", EO)
