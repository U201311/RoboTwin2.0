import zarr 
import os
import shutil
import numpy as np

def main():
    save_dir = f"/workspace/embolab/data_dp/train.zarr"
    zarr_root = zarr.group(save_dir)
    total_count = 0

    if os.path.exists(save_dir):
        zarr_root = zarr.open(save_dir, mode='a')
        zarr_data = zarr_root.require_group('data')
        zarr_meta = zarr_root.require_group('meta')
        # 获取当前的total_count
        if 'episode_ends' in zarr_meta:
            existing_ends = zarr_meta['episode_ends'][:]
            total_count = existing_ends[-1] if len(existing_ends) > 0 else 0
    else:
        zarr_data = zarr_root.create_group("data")
        zarr_meta = zarr_root.create_group("meta")

    # 初始化空列表来收集所有数据
    point_cloud_arrays = []
    state_arrays = []
    joint_action_arrays = []
    episode_ends_arrays = []
    
    load_dir = "/workspace/embolab/data_dp"
    
    # List all zarr files in the directory
    zarr_files = [f for f in os.listdir(load_dir) if f.endswith('.zarr')]
    
    for zarr_file in zarr_files:
        if zarr_file == "train.zarr":
            continue  # Skip the output file if it's in the same directory
        print(f"Processing file: {zarr_file}")
        zarr_path = os.path.join(load_dir, zarr_file)
        zarr_data_in = zarr.open(zarr_path, mode='r')
        
        # 将数据添加到列表中
        point_cloud_arrays.append(zarr_data_in['data/point_cloud'][:])
        print(f"point_cloud length: {len(point_cloud_arrays[-1])}")
        state_arrays.append(zarr_data_in['data/state'][:])
        print(f"state length: {len(state_arrays[-1])}")
        joint_action_arrays.append(zarr_data_in['data/action'][:])

        episode_ends = zarr_data_in['meta/episode_ends'][:]
        episode_ends_arrays.extend(total_count + episode_ends)
        total_count += episode_ends[-1]
    
    # 在所有文件处理完后，一次性写入数据
    if point_cloud_arrays:  # 确保列表不为空
        # 合并所有数据
        point_cloud_combined = np.concatenate(point_cloud_arrays, axis=0)
        state_combined = np.concatenate(state_arrays, axis=0)
        joint_action_combined = np.concatenate(joint_action_arrays, axis=0)
        
        # 确定chunk大小
        point_cloud_chunk_size = (100, point_cloud_combined.shape[1]) if point_cloud_combined.ndim == 2 else (100,)
        state_chunk_size = (100, state_combined.shape[1]) if state_combined.ndim == 2 else (100,)
        joint_chunk_size = (100, joint_action_combined.shape[1]) if joint_action_combined.ndim == 2 else (100,)

        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        
        # 写入数据
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_combined,
            dtype="float32",
            chunks=point_cloud_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        
        zarr_data.create_dataset(
            "state",
            data=state_combined,
            dtype="float32",
            chunks=state_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        
        zarr_data.create_dataset(
            "action",
            data=joint_action_combined,
            dtype="float32",
            chunks=joint_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        
        zarr_meta.create_dataset(
            "episode_ends",
            data=np.array(episode_ends_arrays),
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
        
        print(f"Successfully merged {len(zarr_files)-1} files into {save_dir}")
        print(f"Final dataset sizes:")
        print(f"  point_cloud: {point_cloud_combined.shape}")
        print(f"  state: {state_combined.shape}")
        print(f"  action: {joint_action_combined.shape}")
        print(f"  episode_ends: {len(episode_ends_arrays)} episodes")

if __name__ == "__main__":
    main()