import sys
sys.path.append('..')
from pygdebias.debiasing import EDITS
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
adj, feats, idx_train, idx_val, idx_test, labels, sens = (
    bail.adj(),
    bail.features(),
    bail.idx_train(),
    bail.idx_val(),
    bail.idx_test(),
    bail.labels(),
    bail.sens(),
)

# Initiate the model (with searched parameters).
model = EDITS(
    feats,
).cuda()
model.fit(adj, feats, sens, idx_train, idx_val,
          epochs=1000,
          normalize=True,
          lr=0.003,
          k=-1,
          device="cuda",
          half=True,
          truncation=4,
          )

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
) = model.predict(adj, labels, sens, idx_train, idx_val, idx_test,
                  epochs=1000,
                  lr=0.003,
                  nhid=64,
                  dropout=0.2,
                  weight_decay=1e-7,
                  model="GCN",
                  device="cuda",
                  threshold_proportion=0.015)

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
