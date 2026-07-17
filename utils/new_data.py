import numpy as np
import matplotlib.pyplot as plt

from .jet_dataloader import prepare_constituents
from .jet_utils import pixelize, load_images

def build_image_array(
    images, augment, n_copies, pt_smear, pos_smear, seed, cap=None, normalize=True
):
    """Turn raw jet images into canonicalized 30x30 images.

    When ``augment`` is set, each jet contributes one clean copy plus
    ``n_copies`` smeared copies (smear-then-canonicalize, matching the training
    pipeline). ``normalize`` L1-normalizes each image to sum 1; leaving it off
    keeps the (max-normalized) energy scale, which e.g. ROCKET's max-pooling
    feature can exploit. Returns the image stack and the index of the source
    jet for each row, so labels can be gathered with ``labels[label_idx]``.
    """
    if cap is not None:
        images = images[:cap]
    rng = np.random.default_rng(seed)
    grids, label_idx = [], []
    for i, img in enumerate(images):
        variants = 1 + (n_copies if augment else 0)
        for v in range(variants):
            do_aug = augment and v > 0  # v == 0 is the clean copy
            cons = prepare_constituents(
                img,
                augment=do_aug,
                canonicalize=True,
                pt_smear=pt_smear,
                pos_smear=pos_smear,
                rng=rng,
            )
            grid = pixelize(cons).astype(np.float32)
            if normalize:
                total = grid.sum()
                grid = grid / total if total > 0 else grid
            grids.append(grid)
            label_idx.append(i)
    return np.stack(grids), np.asarray(label_idx)


def generate_new_train(n_copies=5, pt_smear=0.1, pos_smear=0.1):
    """Generate new training data from the original jet images.

    This function loads the original jet images, applies the same preprocessing
    as in the training pipeline, and saves the resulting arrays to disk.
    """
    # Load original images (val/test splits are unused here)
    images, labels, *_ = load_images()

    # Generate new training data
    X_train, label_idx = build_image_array(
        images,
        augment=True,
        n_copies=n_copies,
        pt_smear=pt_smear,
        pos_smear=pos_smear,
        seed=42,
    )
    y_train = labels[label_idx]

    # Save to disk
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)


def total_pt_per_image(images):
    """Total pT of each jet, i.e. the sum of its pixel intensities."""
    return np.asarray(images).reshape(len(images), -1).sum(axis=1)


def accept_reject_mask(total_pt, labels, bins=50, rng=None):
    """Rejection-sample a keep mask that matches the total-pT spectrum across classes.

    Histograms ``total_pt`` per class over shared bins, then thins every class
    down to the rarest class's per-bin count (so no class carries pT values the
    others don't equally share). Within a bin, each event is kept independently
    with probability ``target_count / class_count_in_bin``, giving an unbiased
    random subsample rather than just the first N events.

    Args:
        total_pt (np.ndarray): (N,) total pT per jet.
        labels (np.ndarray): (N,) class label per jet.
        bins (int): Number of shared histogram bins spanning the full pT range.
        rng (np.random.Generator | None): Optional RNG for reproducibility.

    Returns:
        np.ndarray: (N,) boolean mask of jets to keep.
    """
    if rng is None:
        rng = np.random.default_rng()
    total_pt = np.asarray(total_pt)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    edges = np.histogram_bin_edges(total_pt, bins=bins)
    bin_idx = np.clip(np.digitize(total_pt, edges[1:-1]), 0, bins - 1)

    counts = np.stack(
        [np.bincount(bin_idx[labels == c], minlength=bins) for c in classes]
    )
    target = counts.min(axis=0)

    keep = np.zeros(len(total_pt), dtype=bool)
    for ci, c in enumerate(classes):
        for b in range(bins):
            if target[b] == 0:
                continue
            idx = np.where((labels == c) & (bin_idx == b))[0]
            accept_prob = target[b] / counts[ci, b]
            keep[idx[rng.random(len(idx)) < accept_prob]] = True
    return keep


def generate_new_train_accept_reject(
    n_copies=5, pt_smear=0.1, pos_smear=0.1, bins=50, seed=42
):
    """Generate augmented training data with class-balanced total pT.

    Loads the original jets, computes each one's total pT, and rejection-samples
    events so every class shares the same total-pT spectrum -- preventing a
    classifier from learning total energy as a shortcut instead of substructure.
    Augmentation then proceeds as in :func:`generate_new_train`, but only over
    the surviving jets.
    """
    # Load original images (val/test splits are unused here)
    images, labels, *_ = load_images()

    rng = np.random.default_rng(seed)
    total_pt = total_pt_per_image(images)
    keep = accept_reject_mask(total_pt, labels, bins=bins, rng=rng)
    images, labels = images[keep], labels[keep]

    # Generate augmented training data from the reweighted jets
    X_train, label_idx = build_image_array(
        images,
        augment=True,
        n_copies=n_copies,
        pt_smear=pt_smear,
        pos_smear=pos_smear,
        seed=seed,
    )
    y_train = labels[label_idx]

    # Save to disk
    np.save("X_train_ar.npy", X_train)
    np.save("y_train_ar.npy", y_train)


def plot_accept_reject_pt(
    bins=50,
    seed=42,
    class_names=("QCD", "TT"),
    output_path="accept_reject_pt.png",
):
    """Plot per-class total-pT distributions before and after accept-reject.

    Loads the original training jets, computes total pT per jet, and draws a
    two-panel figure: the raw per-class pT spectra (left) next to the spectra
    after :func:`accept_reject_mask` has thinned every class down to the
    rarest class's histogram (right). Saves the figure to ``output_path``.

    Args:
        bins (int): Number of shared histogram bins spanning the full pT range.
        seed (int): Seed for the accept-reject RNG.
        class_names (tuple): Display name for each class, in ``np.unique(labels)`` order.
        output_path (str): Where to save the figure.

    Returns:
        str: The path the figure was saved to.
    """
    # Load original images (val/test splits are unused here)
    images, labels, *_ = load_images()

    total_pt = total_pt_per_image(images)
    rng = np.random.default_rng(seed)
    keep = accept_reject_mask(total_pt, labels, bins=bins, rng=rng)

    edges = np.histogram_bin_edges(total_pt, bins=bins)
    classes = np.unique(labels)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for c, name in zip(classes, class_names):
        ax_before.hist(
            total_pt[labels == c], bins=edges, histtype="step", linewidth=2, label=name
        )
        ax_after.hist(
            total_pt[(labels == c) & keep],
            bins=edges,
            histtype="step",
            linewidth=2,
            label=name,
        )

    ax_before.set_title("Before accept-reject")
    ax_after.set_title("After accept-reject")
    for ax in (ax_before, ax_after):
        ax.set_xlabel("Total pT")
        ax.legend()
    ax_before.set_ylabel("Jets")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path