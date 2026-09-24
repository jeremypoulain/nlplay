"""Anthony Ho, ahho@stanford.edu, 8/4/2016
Last update 8/9/2016
Library functions for embarrassingly parallelization applying functions to large lists,
Pandas series, df (by row), and grouped df"""
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from pandas.api.typing import DataFrameGroupBy, SeriesGroupBy


def _n_jobs(numCores: int) -> int:
    """
    :param numCores: number of workers, negative values follow joblib: -1 → all cores, -2 → all but one.
    :returns: the actual number of workers.
    """
    if not isinstance(numCores, (int, np.integer)) or numCores == 0:
        raise ValueError(f"numCores must be a non zero integer, got {numCores!r}")
    return int(numCores) if numCores > 0 else max(1, cpu_count() + 1 + int(numCores))


def _chunk_bounds(length: int, numChunks: int) -> list[tuple[int, int]]:
    """Split range(length) into at most numChunks contiguous (start, stop) bounds of even size."""
    chunkSize = -(-length // numChunks)
    return [(start, min(start + chunkSize, length)) for start in range(0, length, chunkSize)]


# Apply func to a chunk of a list
def _funcChunkList(dataChunk, func, *args, **kwargs):
    return [func(item, *args, **kwargs) for item in dataChunk]


# Apply func to a chunk of a Pandas series
def _funcChunkSeries(dataChunk, func, *args, **kwargs):
    return dataChunk.apply(func, args=args, **kwargs)


# Apply func to a chunk of a Pandas dataframe
def _funcChunkDf(dataChunk, func, *args, **kwargs):
    return dataChunk.apply(func, axis=1, args=args, **kwargs)


# Apply func to a chunk of a list of (name, group)
def _funcChunkGroups(dataChunk, func, *args, **kwargs):
    return [func(group, *args, **kwargs) for _, group in dataChunk]


def _combine_group_results(data, results: list):
    """
    Rebuild the output of a grouped apply, with the group keys as index.
    func(group) → DataFrame or Series with the group rows: concatenated, prefixed by the group keys
    if group_keys is True, else put back in the original row order.
    func(group) → Series of aggregates: one row per group. Anything else: one value per group.
    """
    # size() gives the group keys index (names, MultiIndex) in the same order as the iteration
    index = data.size().index
    first = results[0]

    if isinstance(first, (pd.DataFrame, pd.Series)) and all(type(r) is type(first) for r in results):
        is_transform = all(r.index.equals(g.index) for r, (_, g) in zip(results, data))
        if isinstance(first, pd.DataFrame) or is_transform:
            if data.group_keys:
                return pd.concat(results, keys=list(index), names=[*index.names, *first.index.names])
            combined = pd.concat(results)
            if is_transform:
                # Transform like results → back to the original row order, by position so that
                # duplicate index labels are supported (ngroup numbers groups in iteration order)
                group_ids = data.ngroup().to_numpy()
                positions = np.concatenate([np.flatnonzero(group_ids == i) for i in range(len(results))])
                combined = combined.iloc[np.argsort(positions, kind="stable")]
            return combined
        return pd.DataFrame(results, index=index)

    return pd.Series(results, index=index)


def parallelApply(data, func, numCores: int, *args, verbose: int = 0, **kwargs):
    """
    Apply a function in parallel to a list, tuple, numpy array, Pandas index, series, dataframe (by row)
    or grouped series / dataframe, the extra args and kwargs are passed to func.
    :param data: input data, chunks are contiguous and keep the input order.
    :param func: function applied to each item, row or group.
    :param numCores: number of workers, negative values follow joblib: -1 → all cores.
    :param verbose: joblib verbosity, keyword only so that extra positional args reach func.
    :returns: a list for list, tuple, array and index inputs, the Pandas equivalent of
        Series.apply, DataFrame.apply(axis=1) or GroupBy.apply otherwise.
    """
    n_jobs = _n_jobs(numCores)
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)

    # Pandas series and dataframe, positional chunks so that duplicate index labels are kept once
    if isinstance(data, (pd.Series, pd.DataFrame)):
        chunkFunc = _funcChunkSeries if isinstance(data, pd.Series) else _funcChunkDf
        if len(data) == 0:
            return chunkFunc(data, func, *args, **kwargs)
        bounds = _chunk_bounds(len(data), n_jobs)
        listResultsChunks = parallel(
            delayed(chunkFunc)(data.iloc[start:stop], func, *args, **kwargs) for start, stop in bounds
        )
        return pd.concat(listResultsChunks)

    # Pandas grouped series or dataframe
    if isinstance(data, (DataFrameGroupBy, SeriesGroupBy)):
        listNamesGroups = list(data)
        if not listNamesGroups:
            raise ValueError("Cannot apply a function to an empty groupby")
        bounds = _chunk_bounds(len(listNamesGroups), n_jobs)
        listResultsChunks = parallel(
            delayed(_funcChunkGroups)(listNamesGroups[start:stop], func, *args, **kwargs)
            for start, stop in bounds
        )
        return _combine_group_results(data, [r for chunk in listResultsChunks for r in chunk])

    # Sequences
    if isinstance(data, (list, tuple, np.ndarray, pd.Index)):
        if len(data) == 0:
            return []
        bounds = _chunk_bounds(len(data), n_jobs)
        listResultsChunks = parallel(
            delayed(_funcChunkList)(data[start:stop], func, *args, **kwargs) for start, stop in bounds
        )
        return [item for sublist in listResultsChunks for item in sublist]

    raise TypeError(f"Unsupported data type for parallelApply: {type(data).__name__}")
