import os
import shutil
import sys
import pathlib
import json
import pytz
import string
import datasets
import numpy as np

from datetime import datetime
from loguru import logger
from typing import Optional
from contextlib import contextmanager
from dataclasses import dataclass



def format_numel_str(numel: int) -> str:
    T = 1e12
    B = 1e9
    M = 1e6
    K = 1e3
    if numel >= T:
        return f"{numel / T:.2f} T"
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def makedirs(path):
    """
    Create all parent dir for path.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path


def rmdirs(directory):
    """Remove all files in the directory."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def split_file_dir_name_ext(path):
    """Return the directory, name, and extension of a given file/directory."""
    p = pathlib.Path(path)
    if p.is_file():
        return p.parent, p.stem, p.suffix
    elif p.is_dir():
        logger.warning(f"{path} is a directory! Return None as suffix!")
        return p.parent, p.name, None
    else:
        raise ValueError(f"{path} not found!")


def get_model_name(model_name_or_path: str, lora: Optional[str]):
    if lora is not None:
        name_or_path_components = [x for x in lora.split("/") if len(x)][-2:]
    else:
        name_or_path_components = [x for x in model_name_or_path.split("/") if len(x)][-2:]
    return os.path.join(*name_or_path_components)


def get_max_length_in_nested_lists(lst):
    if len(lst) and isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)


def pad_nested_lists(lst, max_length, padding_value, padding_side="right"):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        masks = []
        for i, elem in enumerate(lst):
            lst[i], mask = pad_nested_lists(elem, max_length, padding_value, padding_side)
            masks.append(mask)
        return lst, masks
    elif isinstance(lst, list):
        if padding_side == "right":
            mask = [1] * len(lst) + [0] * (max_length - len(lst))
            lst = lst + [padding_value for _ in range(max_length - len(lst))]
            return lst, mask
        else:
            mask = [0] * (max_length - len(lst)) + [1] * len(lst)
            lst = [padding_value for _ in range(max_length - len(lst))] + lst
            return lst, mask
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")


def mask_nested_lists(lst, mask_target, mask_value=0):
    if isinstance(lst[0], list):
        for i, elem in enumerate(lst):
            lst[i] = mask_nested_lists(elem, mask_target, mask_value)
        return lst
    else:
        return [x if x != mask_target else mask_value for x in lst]


def are_elements_of_same_length(lst: list):
    if not isinstance(lst[0], list):
        return False

    length = len(lst[0])
    return all(len(x) == length if isinstance(x, list) else False for x in lst)


def normalize_text(text, ignore_case=True, ignore_punctuation=True, ignore_space=True, ignore_number=False):
    if isinstance(text, str):
        text = [text]
        unpack = True
    else:
        unpack = False
    if ignore_case:
        text = np.char.lower(text)
    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        text = np.char.translate(text, table=repl_table)
    if ignore_number:
        repl_table = string.digits.maketrans("", "", string.digits)
        text = np.char.translate(text, table=repl_table)
    if ignore_space:
        for i, words in enumerate(np.char.split(text)):
            text[i] = " ".join(words)
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if unpack:
        text = text[0]
    return text


# TODO: disable specified transformation
@contextmanager
def dataset_no_transform(dataset: datasets.Dataset):
    """
    Context manager to disable dataset transformation
    """
    if isinstance(dataset, datasets.Dataset):
        dataset_format_transform = dataset.format["format_kwargs"].get("transform", None)
        if dataset_format_transform is not None:
            dataset_format_columns = dataset.format["columns"]
            dataset_format_output_all_columns = dataset.format["output_all_columns"]
            try:
                dataset.reset_format()
                yield dataset
            finally:
                dataset.set_transform(dataset_format_transform, columns=dataset_format_columns, output_all_columns=dataset_format_output_all_columns)
        else:
            yield dataset
    else:
        yield dataset


class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
    
    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone('Asia/Shanghai')
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")
