import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


def plot_tsne_with_kde(
    contras_embeddings_path,
    magpie_embeddings_path,
    reference_embeddings_path,
    output_path="tsne_visualization_with_kde",
):
    contras_embeddings = np.load(contras_embeddings_path)
    magpie_embeddings = np.load(magpie_embeddings_path)
    reference_embeddings = np.load(reference_embeddings_path)

    embeddings = np.vstack((contras_embeddings, magpie_embeddings, reference_embeddings))
    labels = np.array([0] * contras_embeddings.shape[0] + [1] * magpie_embeddings.shape[0] + [2] * reference_embeddings.shape[0])

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    contras_2d = embeddings_2d[labels == 0]
    magpie_2d = embeddings_2d[labels == 1]
    reference_2d = embeddings_2d[labels == 2]

    x_min, x_max = embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1
    y_min, y_max = embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_points = np.vstack([xx.ravel(), yy.ravel()])

    kde_contras = gaussian_kde(contras_2d.T)
    kde_magpie = gaussian_kde(magpie_2d.T)
    kde_reference = gaussian_kde(reference_2d.T)

    z_contras = kde_contras(grid_points).reshape(xx.shape)
    z_magpie = kde_magpie(grid_points).reshape(xx.shape)
    z_reference = kde_reference(grid_points).reshape(xx.shape)

    colors = ['#1f77b4', '#f3d19c', '#E4080A']
    labels_text = ["DOMINO", "MAGPIE-Few Shot", "Reference"]
    datasets = [(contras_2d, z_contras, colors[0]), (magpie_2d, z_magpie, colors[1]), (reference_2d, z_reference, colors[2])]

    plt.figure(figsize=(8, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    for i, (data_2d, z, color) in enumerate(datasets):
        plt.contourf(xx, yy, z, levels=8, cmap=ListedColormap([color]), alpha=0.2)
        contour = plt.contour(xx, yy, z, levels=5, colors=color, linewidths=1)
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=color, label=labels_text[i], s=10, alpha=0.4, edgecolor='w')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=labels_text[i],
               markerfacecolor=colors[i], markersize=10)
        for i in range(3)
    ]
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=16, frameon=False)

    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_path}.pdf", format="pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_path}.png", format="png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--contras_embeddings", type=str, required=True)
    parser.add_argument("--magpie_embeddings", type=str, required=True)
    parser.add_argument("--reference_embeddings", type=str, required=True)
    parser.add_argument("--output", type=str, default="tsne_visualization_with_kde")
    args = parser.parse_args()

    plot_tsne_with_kde(
        args.contras_embeddings,
        args.magpie_embeddings,
        args.reference_embeddings,
        args.output,
    )
