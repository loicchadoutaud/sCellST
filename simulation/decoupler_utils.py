"""
Imported from https://github.com/saezlab/decoupler-py/blob/main/decoupler/utils_anndata.py

Utility functions for AnnData objects.
Functions to process AnnData objects.
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
import pandas as pd

from anndata import AnnData


def swap_layer(adata, layer_key, X_layer_key="X", inplace=False):
    """
    Swaps an ``adata.X`` for a given layer.

    Swaps an AnnData ``X`` matrix with a given layer. Generates a new object by default.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer_key : str
        ``.layers`` key to place in ``.X``.
    X_layer_key : str, None
        ``.layers`` key where to move and store the original ``.X``. If None, the original ``.X`` is discarded.
    inplace : bool
        If ``False``, return a copy. Otherwise, do operation inplace and return ``None``.

    Returns
    -------
    layer : AnnData, None
        If ``inplace=False``, new AnnData object.
    """

    cdata = None
    if inplace:
        if X_layer_key is not None:
            adata.layers[X_layer_key] = adata.X
        adata.X = adata.layers[layer_key]
    else:
        cdata = adata.copy()
        if X_layer_key is not None:
            cdata.layers[X_layer_key] = cdata.X
        cdata.X = cdata.layers[layer_key]

    return cdata


def extract_psbulk_inputs(adata, obs, layer, use_raw):

    # Extract count matrix X
    if layer is not None:
        X = adata.layers[layer]
    elif type(adata) is AnnData:
        if use_raw:
            if adata.raw is None:
                raise ValueError("Received `use_raw=True`, but `mat.raw` is empty.")
            X = adata.raw.X
        else:
            X = adata.X
    else:
        X = adata.values

    # Extract meta-data
    if type(adata) is AnnData:
        obs = adata.obs
        var = adata.var
    else:
        var = pd.DataFrame(index=adata.columns)
        if obs is None:
            raise ValueError("If adata is a pd.DataFrame, obs cannot be None.")

        # Match indexes of X with obs if DataFrame
        idxs = adata.index
        try:
            obs = obs.loc[idxs]
        except KeyError:
            raise KeyError("Indices in obs do not match with mat's.")

    # Sort genes
    msk = np.argsort(var.index)
    X = X[:, msk]
    var = var.iloc[msk]

    if issparse(X) and not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    return X, obs, var


def check_X(X, mode="sum", skip_checks=False):
    if isinstance(X, csr_matrix):
        is_finite = np.all(np.isfinite(X.data))
    else:
        is_finite = np.all(np.isfinite(X))
    if not is_finite:
        raise ValueError(
            "Data contains non finite values (nan or inf), please set them to 0 or remove them."
        )
    skip_checks = type(mode) is dict or callable(mode) or skip_checks
    if not skip_checks:
        if isinstance(X, csr_matrix):
            is_positive = np.all(X.data >= 0)
        else:
            is_positive = np.all(X >= 0)
        if not is_positive:
            raise ValueError(
                """Data contains negative values. Check the parameters use_raw and layers to
            determine if you are selecting the correct matrix. To override this, set skip_checks=True.
            """
            )
        if mode == "sum":
            if isinstance(X, csr_matrix):
                is_integer = float(np.sum(X.data)).is_integer()
            else:
                is_integer = float(np.sum(X)).is_integer()
            if not is_integer:
                raise ValueError(
                    """Data contains float (decimal) values. Check the parameters use_raw and layers to
                determine if you are selecting the correct data, which should be positive integer counts when mode='sum'.
                To override this, set skip_checks=True.
                """
                )


def format_psbulk_inputs(sample_col, groups_col, obs):
    # Use one column if the same
    if sample_col == groups_col:
        groups_col = None

    if groups_col is None:
        # Filter extra columns in obs
        cols = obs.groupby(sample_col, observed=True).nunique(dropna=False).eq(1).all(0)
        cols = np.hstack([sample_col, cols[cols].index])
        obs = obs.loc[:, cols]

        # Get unique samples
        smples = np.unique(obs[sample_col].values)
        groups = None

        # Get number of samples and features
        n_rows = len(smples)
    else:
        # Check if extra grouping is needed
        if type(groups_col) is list:
            obs = obs.copy()
            joined_cols = "_".join(groups_col)
            obs[joined_cols] = obs[groups_col[0]].str.cat(
                obs[groups_col[1:]].astype("U"), sep="_"
            )
            groups_col = joined_cols

        # Filter extra columns in obs
        cols = (
            obs.groupby([sample_col, groups_col], observed=True)
            .nunique(dropna=False)
            .eq(1)
            .all(0)
        )
        cols = np.hstack([sample_col, groups_col, cols[cols].index])
        obs = obs.loc[:, cols]

        # Get unique samples and groups
        smples = np.unique(obs[sample_col].values)
        groups = np.unique(obs[groups_col].values)

        # Get number of samples and features
        n_rows = len(smples) * len(groups)

    return obs, groups_col, smples, groups, n_rows


def psbulk_profile(profile, mode="sum"):
    if mode == "sum":
        profile = np.sum(profile, axis=0)
    elif mode == "mean":
        profile = np.mean(profile, axis=0)
    elif mode == "median":
        profile = np.median(profile, axis=0)
    elif callable(mode):
        profile = np.apply_along_axis(mode, 0, profile)
    else:
        raise ValueError(
            """mode={0} can be 'sum', 'mean', 'median' or a callable function.""".format(
                mode
            )
        )
    return profile


def compute_psbulk(
    n_rows,
    n_cols,
    X,
    sample_col,
    groups_col,
    smples,
    groups,
    obs,
    new_obs,
    min_cells,
    min_counts,
    mode,
    dtype,
):

    # Init empty variables
    psbulk = np.zeros((n_rows, n_cols))
    props = np.zeros((n_rows, n_cols))
    ncells = np.zeros(n_rows)
    counts = np.zeros(n_rows)

    # Iterate for each group and sample
    i = 0
    if groups_col is None:
        for smp in smples:
            # Write new meta-data
            tmp = obs[obs[sample_col] == smp].drop_duplicates().values
            new_obs.loc[smp, :] = tmp

            # Get cells from specific sample
            profile = X[obs[sample_col] == smp]
            if isinstance(X, csr_matrix):
                profile = profile.toarray()

            # Skip if few cells or not enough counts
            ncell = profile.shape[0]
            count = np.sum(profile)
            ncells[i] = ncell
            counts[i] = count
            if ncell < min_cells or np.abs(count) < min_counts:
                i += 1
                continue

            # Get prop of non zeros
            prop = np.sum(profile != 0, axis=0) / profile.shape[0]

            # Pseudo-bulk
            profile = psbulk_profile(profile, mode=mode)

            # Append
            props[i] = prop
            psbulk[i] = profile
            i += 1
    else:
        for grp in groups:
            for smp in smples:

                # Write new meta-data
                index = smp + "_" + grp
                tmp = (
                    obs[(obs[sample_col] == smp) & (obs[groups_col] == grp)]
                    .drop_duplicates()
                    .values
                )
                if tmp.shape[0] == 0:
                    tmp = np.full(tmp.shape[1], np.nan)
                new_obs.loc[index, :] = tmp

                # Get cells from specific sample and group
                profile = X[(obs[sample_col] == smp) & (obs[groups_col] == grp)]
                if isinstance(X, csr_matrix):
                    profile = profile.toarray()

                # Skip if few cells or not enough counts
                ncell = profile.shape[0]
                count = np.sum(profile)
                ncells[i] = ncell
                counts[i] = count
                if ncell < min_cells or np.abs(count) < min_counts:
                    i += 1
                    continue

                # Get prop of non zeros
                prop = np.sum(profile != 0, axis=0) / profile.shape[0]

                # Pseudo-bulk
                profile = psbulk_profile(profile, mode=mode)

                # Append
                props[i] = prop
                psbulk[i] = profile
                i += 1

    return psbulk, ncells, counts, props


def get_pseudobulk(
    adata,
    sample_col,
    groups_col,
    obs=None,
    layer=None,
    use_raw=False,
    mode="sum",
    min_cells=10,
    min_counts=1000,
    dtype=np.float32,
    skip_checks=False,
    min_prop=None,
    min_smpls=None,
    remove_empty=True,
):
    """
    Summarizes expression profiles across cells per sample and group.

    Generates summarized expression profiles across cells per sample (e.g. sample id) and group (e.g. cell type) based on the
    metadata found in ``.obs``. To ensure a minimum quality control, this function removes genes that are not expressed enough
    across cells (``min_prop``) or samples (``min_smpls``), and samples with not enough cells (``min_cells``) or gene counts
    (``min_counts``).

    By default this function expects raw integer counts as input and sums them per sample and group (``mode='sum'``), but other
    modes are available.

    This function produces some quality control metrics to assess if is necessary to filter some samples. The number of cells
    that belong to each sample is stored in ``.obs['psbulk_n_cells']``, the total sum of counts per sample in
    ``.obs['psbulk_counts']``, and the proportion of cells that express a given gene in ``.layers['psbulk_props']``.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    sample_col : str
        Column of `obs` where to extract the samples names.
    groups_col : str
        Column of `obs` where to extract the groups names. Can be set to ``None`` to ignore groups.
    obs : DataFrame, None
        If provided, metadata dataframe.
    layer : str
        If provided, which element of layers to use.
    use_raw : bool
        Use `raw` attribute of `adata` if present.
    mode : str
        How to perform the pseudobulk. Available options are ``sum``, ``mean`` or ``median``. It also accepts callback
        functions, like lambda, to perform custom aggregations. Additionally, it is also possible to provide a dictionary of
        different callback functions, each one stored in a different resulting `.layer`. In this case, the result of the first
        callback function of the dictionary is stored in ``.X`` by default. To switch between layers check
        ``decoupler.swap_layer``.
    min_cells : int
        Filter to remove samples by a minimum number of cells in a sample-group pair.
    min_counts : int
        Filter to remove samples by a minimum number of summed counts in a sample-group pair.
    dtype : type
        Type of float used.
    skip_checks : bool
        Whether to skip input checks. Set to ``True`` when working with positive and negative data, or when counts are not
        integers.
    min_prop : float
        Filter to remove features by a minimum proportion of cells with non-zero values. Deprecated parameter,
        check ``decoupler.filter_by_prop``.
    min_smpls : int
        Filter to remove genes by a minimum number of samples with non-zero values. Deprecated parameter,
        check ``decoupler.filter_by_prop``.
    remove_empty : bool
        Whether to remove empty observations (rows) or features (columns).

    Returns
    -------
    psbulk : AnnData
        Returns new AnnData object with unormalized pseudobulk profiles per sample and group. It also returns quality control
        metrics that start with the prefix ``psbulk_``.
    """

    min_cells, min_counts = np.clip(min_cells, 1, None), np.clip(min_counts, 1, None)

    # Extract inputs
    X, obs, var = extract_psbulk_inputs(adata, obs, layer, use_raw)

    # Test if X is correct
    check_X(X, mode=mode, skip_checks=skip_checks)

    # Format inputs
    obs, groups_col, smples, groups, n_rows = format_psbulk_inputs(
        sample_col, groups_col, obs
    )
    n_cols = adata.shape[1]
    new_obs = pd.DataFrame(columns=obs.columns)

    if type(mode) is dict:
        psbulks = []
        for l_name in mode:
            func = mode[l_name]
            if not callable(func):
                raise ValueError(
                    """mode requieres a dictionary of layer names and callable functions. The layer {0} does not
                contain one.""".format(
                        l_name
                    )
                )
            else:
                # Compute psbulk
                psbulk, ncells, counts, props = compute_psbulk(
                    n_rows,
                    n_cols,
                    X,
                    sample_col,
                    groups_col,
                    smples,
                    groups,
                    obs,
                    new_obs,
                    min_cells,
                    min_counts,
                    func,
                    dtype,
                )
                psbulks.append(psbulk)
        layers = {k: v for k, v in zip(mode.keys(), psbulks)}
        layers["psbulk_props"] = props
    elif type(mode) is str or callable(mode):
        # Compute psbulk
        psbulk, ncells, counts, props = compute_psbulk(
            n_rows,
            n_cols,
            X,
            sample_col,
            groups_col,
            smples,
            groups,
            obs,
            new_obs,
            min_cells,
            min_counts,
            mode,
            dtype,
        )
        layers = {"psbulk_props": props}

    # Add QC metrics
    new_obs["psbulk_n_cells"] = ncells
    new_obs["psbulk_counts"] = counts

    # Create new AnnData
    psbulk = AnnData(psbulk.astype(dtype), obs=new_obs, var=var, layers=layers)

    # Remove empty samples and features
    if remove_empty:
        msk = psbulk.X == 0
        psbulk = psbulk[~np.all(msk, axis=1), ~np.all(msk, axis=0)].copy()

    # Place first element of mode dict as X
    if type(mode) is dict:
        swap_layer(
            psbulk, layer_key=list(mode.keys())[0], X_layer_key=None, inplace=True
        )

    # Filter by genes if not None.
    if min_prop is not None and min_smpls is not None:
        if groups_col is None:
            genes = filter_by_prop(psbulk, min_prop=min_prop, min_smpls=min_smpls)
        else:
            genes = []
            for group in groups:
                g = filter_by_prop(
                    psbulk[psbulk.obs[groups_col] == group],
                    min_prop=min_prop,
                    min_smpls=min_smpls,
                )
                genes.extend(g)
            genes = np.unique(genes)
        psbulk = psbulk[:, genes]

    return psbulk


def filter_by_prop(adata, min_prop=0.2, min_smpls=2):
    """
    Determine which genes are expressed in a sufficient proportion of cells across samples.

    This function selects genes that are sufficiently expressed across cells in each sample and that this condition is
    met across a minimum number of samples.

    Parameters
    ----------
    adata : AnnData
        AnnData obtained after running ``decoupler.get_pseudobulk``. It requires ``.layer['psbulk_props']``.
    min_prop : float
        Minimum proportion of cells that express a gene in a sample.
    min_smpls : int
        Minimum number of samples with bigger or equal proportion of cells with expression than ``min_prop``.

    Returns
    -------
    genes : ndarray
        List of genes to be kept.
    """

    # Define limits
    min_prop = np.clip(min_prop, 0, 1)
    min_smpls = np.clip(min_smpls, 0, adata.shape[0])

    if isinstance(adata, AnnData):
        layer_keys = adata.layers.keys()
        if "psbulk_props" in list(layer_keys):
            var_names = adata.var_names.values.astype("U")
            props = adata.layers["psbulk_props"]
            if isinstance(props, pd.DataFrame):
                props = props.values

            # Compute n_smpl
            nsmpls = np.sum(props >= min_prop, axis=0)

            # Set features to 0
            msk = nsmpls >= min_smpls
            genes = var_names[msk]
            return genes
    raise ValueError(
        """adata must be an AnnData object that contains the layer 'psbulk_props'. Please check the function
                     decoupler.get_pseudobulk."""
    )
