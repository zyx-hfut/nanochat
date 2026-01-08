"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

CN_DATA_DIR = "/path/to/your/chinese_data"  # 中文文件夹路径
EN_DATA_DIR = "/path/to/your/english_data"  # 英文文件夹路径

SHUFFLE_SEED = 42




# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported
def _get_sorted_files_from_dir(directory):
    """(Internal helper) Get sorted list of parquet files from a directory."""
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return []
    files = sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return files
# def list_parquet_files(data_dir=None):
#     """ Looks into a data dir and returns full paths to all parquet files. """
#     data_dir = DATA_DIR if data_dir is None else data_dir
#     parquet_files = sorted([
#         f for f in os.listdir(data_dir)
#         if f.endswith('.parquet') and not f.endswith('.tmp')
#     ])
#     parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
#     return parquet_paths
def list_parquet_files(data_dir=None):
    """ 
    Looks into data dirs and returns full paths to all parquet files. 
    如果未指定 data_dir，默认返回中英文文件夹下所有的文件列表（合并）。
    """
    if data_dir is not None:
        # 如果外部调用指定了特定文件夹，保持原有行为
        return _get_sorted_files_from_dir(data_dir)
    
    # 默认情况：返回所有中文和英文文件的总和
    cn_files = _get_sorted_files_from_dir(CN_DATA_DIR)
    en_files = _get_sorted_files_from_dir(EN_DATA_DIR)
    return cn_files + en_files

# def parquets_iter_batched(split, start=0, step=1):
#     """
#     Iterate through the dataset, in batches of underlying row_groups for efficiency.
#     - split can be "train" or "val". the last parquet file will be val.
#     - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
#     """
#     assert split in ["train", "val"], "split must be 'train' or 'val'"
#     parquet_paths = list_parquet_files()
#     parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
#     for filepath in parquet_paths:
#         pf = pq.ParquetFile(filepath)
#         for rg_idx in range(start, pf.num_row_groups, step):
#             rg = pf.read_row_group(rg_idx)
#             texts = rg.column('text').to_pylist()
#             yield texts
def parquets_iter_batched(split, start=0, step=1):
    """
    【保持函数名不变】
    Iterate through the dataset.
    - split: "train" or "val"
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    # 1. 获取两组文件
    cn_files = _get_sorted_files_from_dir(CN_DATA_DIR)
    en_files = _get_sorted_files_from_dir(EN_DATA_DIR)
    
    # 确保文件存在
    if not cn_files or not en_files:
        raise ValueError(f"Data missing. Found {len(cn_files)} CN files and {len(en_files)} EN files.")

    # 2. 【核心逻辑修改】自定义划分策略
    # 验证集：强制取中文最后一个 + 英文最后一个
    val_files = [cn_files[-1], en_files[-1]]
    
    # 训练集：取各自剩余的文件
    train_cn = cn_files[:-1]
    train_en = en_files[:-1]
    
    if split == "val":
        parquet_paths = val_files
    else: # train
        # 合并
        parquet_paths = train_cn + train_en
        # 混合：使用固定种子打乱，实现中英文交叉读取
        # 注意：这里必须用固定种子，否则多卡训练(DDP)时不同显卡读取顺序不一致会报错
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(parquet_paths)

    # 打印信息用于调试
    if start == 0: # 只在主进程打印
        print(f"Dataset split '{split}': loading {len(parquet_paths)} files.")
        if split == "train":
            # 打印前几个文件名，确认混合情况
            sample_names = [os.path.basename(f) for f in parquet_paths[:3]]
            print(f"  (Train shuffle sample: {sample_names} ...)")

    # 3. 迭代读取 (保持原有逻辑)
    for filepath in parquet_paths:
        try:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                # 兼容性处理：尝试获取 'text' 列
                if 'text' in rg.column_names:
                    yield rg.column('text').to_pylist()
                # 如果你的数据列名是 content 或 body，可以在这里添加 fallback
                elif 'content' in rg.column_names:
                    yield rg.column('content').to_pylist()
                else:
                    print("WARNING: WRONG COLUMN NAMES(NOT IN 'text' or 'content')")
                    # pass # 或者打印警告
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=8, help="Number of parallel download workers (default: 8)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
