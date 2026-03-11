from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

dataset_path = Path("dataset")
dataset_path.mkdir(
    parents=True,
    exist_ok=True,
)
output_dir = "."

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
)

dataset = ImageFolder(
    root=dataset_path,
    transform=transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)

train_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=0)

resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1,
                  progress=True
)


resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.to(device)
print(resnet)

z = []
y = []
with torch.no_grad():
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        features = resnet(x_batch)
        z.extend(features.view(features.size(0), -1).cpu().numpy())
        y.extend(y_batch.cpu().numpy())

z = np.stack(z).astype(np.float64)  # size: n x d
y = np.array(y)
n, d = z.shape
print("n, d: ", n, d)


def eigens_and_rank(ZZ):
    _, eigs, _ = np.linalg.svd(ZZ, full_matrices=False)
    r = np.linalg.matrix_rank(ZZ)

    return eigs, r


def pre_transrate_low_dim_proj(f, y):
    n, d = f.shape
    Z = f

    mean_f = np.mean(f, axis=0)
    mean_f = np.expand_dims(mean_f, 1)

    covf = f.T @ f / n - mean_f * mean_f.T

    K = int(y.max() + 1)

    eig_c = []
    nc = []
    rank_c = []
    cov_Zi = []

    for i in range(K):
        y_ = (y == i).flatten()
        Zi = Z[y_]
        nci = Zi.shape[0]
        nc.append(nci)

        mean_Zi = np.mean(Zi, axis=0)
        mean_Zi = np.expand_dims(mean_Zi, 1)
        g = mean_Zi - mean_f

        if i == 0:
            covg = g @ g.T * nci
        else:
            covg = g @ g.T * nci + covg

        ZZi = Zi.T @ Zi - (mean_Zi * mean_Zi.T) * nci

        cov_Zi.append(ZZi)

    proj_mat = np.dot(np.linalg.pinv(covf, rcond=1e-15), covg/n)


    for i in range(K):
        eigs_i, ri = eigens_and_rank(cov_Zi[i] @ proj_mat / nc[i])
        eig_c.extend(np.expand_dims(eigs_i, axis=0))
        rank_c.append(ri)
    eig_Z, rank_Z = eigens_and_rank(covf @ proj_mat)

    return eig_Z, rank_Z, np.stack(eig_c), rank_c, nc


def transrate_eig_proj(z, y):
    eig_Z, rank_Z, eig_Zc, rank_Zc, n_Zc = pre_transrate_low_dim_proj(np.copy(z), np.copy(y))
    data = {
        'eig_Z': eig_Z,
        'rank_Z': rank_Z,
        'eig_Zc': eig_Zc,
        'rank_Zc': rank_Zc,
        'n_Zc': n_Zc
    }

    return data


def transrate_from_eigs(eigs, eps=1e-12):
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = np.clip(eigs, eps, None)          
    p = eigs / eigs.sum()                    
    H = -np.sum(p * np.log(p))               
    return float(np.exp(H))                  

def compute_transrate_scores(z, y):
    out = transrate_eig_proj(np.copy(z), np.copy(y))

    tr_global = transrate_from_eigs(out["eig_Z"])
    tr_class = np.array([transrate_from_eigs(e) for e in out["eig_Zc"]], dtype=np.float64)

    n_Zc = np.array(out["n_Zc"], dtype=np.float64)
    tr_class_weighted = float(np.sum(tr_class * (n_Zc / n_Zc.sum())))

    tr_gap = tr_global - tr_class_weighted

    return {
        **out,
        "transrate_global": tr_global,
        "transrate_class": tr_class,
        "transrate_class_weighted": tr_class_weighted,
        "transrate_gap": tr_gap,
    }

scores = compute_transrate_scores(z, y)
print("transrate_global:", scores["transrate_global"])
print("transrate_class_weighted:", scores["transrate_class_weighted"])
print("transrate_gap:", scores["transrate_gap"])

