from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import multivariate_normal

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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2, 
    perplexity=10, 
    learning_rate='auto', 
    random_state=1410
)
z_embedded = tsne.fit_transform(z)

class_names = dataset.classes
labels_named = [class_names[i] for i in y]

df_tsne = pd.DataFrame({
    "t-SNE 1": z_embedded[:, 0],
    "t-SNE 2": z_embedded[:, 1],
    "Class": labels_named
})

plt.figure(figsize=(10, 8))
sns.set_style("white")

sns.scatterplot(
    data=df_tsne,
    x="t-SNE 1",
    y="t-SNE 2",
    hue="Class",
    palette="viridis",
    alpha=0.6,
    s=40,
    edgecolor='none'
)

plt.title(f"t-SNE Visualization of Deep Features (ResNet-18)", fontsize=14)
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend(title="Target Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("feature_visualization_tsne.png", dpi=300)
plt.show()


indices = np.array(dataset.targets)
drone_idx = dataset.class_to_idx['drone']
bird_idx = dataset.class_to_idx['bird']

z_id = z[indices == drone_idx]  
z_ood = z[indices == bird_idx]  

pca = PCA(n_components=32) 
z_id_train = pca.fit_transform(z_id)
z_ood_test = pca.transform(z_ood)


density_model = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
density_model.fit(z_id_train)


scores_id = density_model.score_samples(z_id_train)
scores_ood = density_model.score_samples(z_ood_test)


y_true = np.concatenate([np.ones(len(scores_id)), np.zeros(len(scores_ood))])
y_scores = np.concatenate([scores_id, scores_ood])

auroc = roc_auc_score(y_true, y_scores)
aupr = average_precision_score(y_true, y_scores)

print(f"AUROC: {auroc:.4f}")
print(f"AUPR:  {aupr:.4f}")


