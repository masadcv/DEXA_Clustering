import argparse
import os

import numpy as np


def main(args):
    dexa_embeddings = np.load(
        os.path.join(args.dataset_path, "ukb_dexa_embeddings.npy")
    )
    labels = np.load(os.path.join(args.dataset_path, "kmeans_labels.npy"))

    print(dexa_embeddings.shape)
    print(labels.shape)

    print("Running t-SNE...")
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=600)
    tsne_results = tsne.fit_transform(dexa_embeddings)
    print("t-SNE done!")
    print("Plotting...")

    # visualize gif showing the 3D t-SNE plot with the clusters colored
    import plotly.express as px

    df = {
        "tsne-2d-one": tsne_results[:, 0],
        "tsne-2d-two": tsne_results[:, 1],
        "tsne-2d-three": tsne_results[:, 2],
        "label": labels,
    }
    fig = px.scatter_3d(
        df,
        x="tsne-2d-one",
        y="tsne-2d-two",
        z="tsne-2d-three",
        color="label",
        title="t-SNE 3D",
    )

    fig.write_html(os.path.join(args.output, "3d_tsne.html"))
    print("Done!")

    # create gif with rotating 3D plot for powerpoint
    # save gif at the end
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=tsne_results[:, 0],
                y=tsne_results[:, 1],
                z=tsne_results[:, 2],
                mode="markers",
                marker=dict(color=labels, size=5),
            )
        ]
    )

    frames = []
    for i in range(0, 360, 2):
        fig.update_layout(scene_camera=dict(eye=dict(x=0, y=0, z=0)))
        frames.append(go.Frame(layout=fig.layout))

    fig.frames = frames
    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False)])

    fig.write_html(os.path.join(args.output, "3d_tsne_rotating.html"))
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/data/Coding/ImageClusteringDEXA/Datasets/output_4clusters_5000images/",
        type=str,
    )
    parser.add_argument("--output", default="./output_3d_tsne", type=str)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    main(args)
