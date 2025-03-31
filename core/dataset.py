import zarr
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

Q_LEARNING_KEYS = {
    "nobs": ("next_observations", True),
    "rewards": ("rewards", False),
    "is_terminal": ("is_terminal", False),
}


def _flatten_if_needed(array: NDArray):
    if array.ndim == 3:
        return array.reshape(-1, array.shape[-1] * array.shape[-2])  # type: ignore
    return array


def create_sample_indices(
    episode_ends: NDArray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    ndata = data.copy()
    for i in range(data.shape[1]):
        if stats["max"][i] - stats["min"][i] > 0:
            ndata[:, i] = (data[:, i] - stats["min"][i]) / (
                stats["max"][i] - stats["min"][i]
            )
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


class MultiArmDataset(Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):

        dataset = zarr.open(dataset_path, mode="r")
        if hasattr(dataset, "array_keys"):
            available_arrays = set(dataset.array_keys())  # type: ignore
        else:
            available_arrays = set(dataset.keys())  # type: ignore

        train_data = {
            "obs": _flatten_if_needed(np.asarray(dataset["observations"][:])),
            "actions": dataset["actions"][:],
        }

        for key, (dataset_key, should_flatten) in Q_LEARNING_KEYS.items():
            if dataset_key not in available_arrays:
                continue
            value = dataset[dataset_key][:]
            if should_flatten:
                value = _flatten_if_needed(np.asarray(value))
            train_data[key] = value

        indices = create_sample_indices(
            episode_ends=np.array(dataset["episode_ends"][:]),
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            if key not in ["rewards", "is_terminal"]:
                stats[key] = get_data_stats(data)
                normalized_train_data[key] = normalize_data(data, stats[key])
            else:
                normalized_train_data[key] = data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        if "nobs" in nsample:
            nsample["nobs"] = nsample["nobs"][: self.obs_horizon, :]
        nsample["idx"] = idx
        return nsample
