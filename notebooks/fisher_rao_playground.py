import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from fisher_rao_ml.losses import (
        categorical_fisher_rao_squared,
        diagonal_gaussian_fisher_rao_squared,
    )

    return categorical_fisher_rao_squared, diagonal_gaussian_fisher_rao_squared, mo, plt, torch


@app.cell
def _(mo):
    mo.md(
        """
        # Fisher-Rao ML Playground

        This notebook checks the core objective pieces before running full experiments:

        - categorical Fisher-Rao on probability vectors, for t-SNE-style matching;
        - diagonal Gaussian Fisher-Rao to the standard normal, for VAE regularization.
        """
    )
    return


@app.cell
def _(categorical_fisher_rao_squared, plt, torch):
    t = torch.linspace(0.01, 0.99, 200)
    p = torch.stack([t, 1.0 - t], dim=-1)
    q = torch.tensor([0.5, 0.5]).expand_as(p)
    fr = categorical_fisher_rao_squared(p, q).detach()
    kl = (p * (p.log() - q.log())).sum(dim=-1).detach()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t.numpy(), fr.numpy(), label="Fisher-Rao squared")
    ax.plot(t.numpy(), kl.numpy(), label="KL")
    ax.set_xlabel("p(class 0)")
    ax.set_ylabel("divergence / distance")
    ax.legend()
    ax.set_title("Categorical objective shape")
    return fig


@app.cell
def _(diagonal_gaussian_fisher_rao_squared, mo, torch):
    mean = torch.tensor([[0.0, 0.5, 1.0]], requires_grad=True)
    logvar = torch.zeros_like(mean, requires_grad=True)
    loss = diagonal_gaussian_fisher_rao_squared(mean, logvar).mean()
    loss.backward()

    mo.md(
        f"""
        ## Gradient sanity check

        Gaussian Fisher-Rao squared loss: `{float(loss.detach()):.4f}`

        Gradient wrt mean: `{mean.grad.detach().numpy().round(4).tolist()}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
