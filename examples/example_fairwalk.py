import sys
sys.path.append('..')
from pygdebias.debiasing import FairWalk
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
model = FairWalk()
model.fit(adj,  labels, idx_train, sens)


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
) = model.predict(idx_test, idx_val)

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
