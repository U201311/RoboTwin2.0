import zarr 
import os
import shutil
import numpy as np




def main():
    save_dir = f"/workspace/embolab/data_dp/train.zarr"
    zarr_root = zarr.group(save_dir)
    total_count = 0
    point_cloud_arrays = []
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = (
        [],
        [],
        [],
        [],
    )

    if os.path.exists(save_dir):
        zarr_root = zarr.open(save_dir, mode='a')
        zarr_data = zarr_root.require_group('data')
        zarr_meta = zarr_root.require_group('meta')
        # 获取当前的total_count
        if 'episode_ends' in zarr_meta:
            existing_ends = zarr_meta['episode_ends'][:]
            total_count = existing_ends[-1] if len(existing_ends) > 0 else 0
        # 安全读取数据
        point_cloud_arrays = zarr_data['point_cloud'][:] if 'point_cloud' in zarr_data else []
        state_arrays = zarr_data['state'][:] if 'state' in zarr_data else []
        joint_action_arrays = zarr_data['action'][:] if 'action' in zarr_data else []
        episode_ends_arrays = zarr_meta['episode_ends'][:] if 'episode_ends' in zarr_meta else []
    else:
        zarr_data = zarr_root.create_group("data")
        zarr_meta = zarr_root.create_group("meta")

        
    load_dir = "/workspace/embolab/data_dp"
    
    # List all zarr files in the directory
    zarr_files = [f for f in os.listdir(load_dir) if f.endswith('.zarr')]
    

    
    for zarr_file in zarr_files:
        if zarr_file == "train.zarr":
            continue  # Skip the output file if it's in the same directory
        print(f"Processing file: {zarr_file}")
        zarr_path = os.path.join(load_dir, zarr_file)
        zarr_data_in = zarr.open(zarr_path, mode='r')
        
        point_cloud_arrays.append(zarr_data_in['data/point_cloud'][:])
        print(f"poiny_cloud length: {len(point_cloud_arrays[-1])}")
        state_arrays.append(zarr_data_in['data/state'][:])
        print(f"state length: {len(state_arrays[-1])}")
        joint_action_arrays.append(zarr_data_in['data/action'][:])

        # 合并为numpy数组后再取shape
        point_cloud_np = np.concatenate(point_cloud_arrays, axis=0) if len(point_cloud_arrays) > 0 else np.empty((0,))
        state_np = np.concatenate(state_arrays, axis=0) if len(state_arrays) > 0 else np.empty((0,))
        joint_action_np = np.concatenate(joint_action_arrays, axis=0) if len(joint_action_arrays) > 0 else np.empty((0,))

        point_cloud_chunk_size = (100, point_cloud_np.shape[1]) if point_cloud_np.ndim == 2 else (100,)
        state_chunk_size = (100, state_np.shape[1]) if state_np.ndim == 2 else (100,)
        joint_chunk_size = (100, joint_action_np.shape[1]) if joint_action_np.ndim == 2 else (100,)

        episode_ends = zarr_data_in['meta/episode_ends'][:]
        episode_ends_arrays.extend(total_count + episode_ends)
        total_count += episode_ends[-1]
        
        #按照chunk_size 写入数据
        
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        zarr_data.create_dataset(
            "point_cloud",
            data=np.concatenate(point_cloud_arrays, axis=0),
            dtype="float32",
            chunks=point_cloud_chunk_size,
            overwrite=True,
            
            compressor=compressor,
        )
        
        zarr_data.create_dataset(
            "state",
            data=np.concatenate(state_arrays, axis=0),
            dtype="float32",
            chunks=state_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "action",
            data=np.concatenate(joint_action_arrays, axis=0),
            dtype="float32",
            chunks=joint_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        # 兼容 action/joint_action 命名
        zarr_data.create_dataset(
            "action",
            data=np.concatenate(joint_action_arrays, axis=0),
            dtype="float32",
            chunks=joint_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        
        zarr_meta.create_dataset(
            "episode_ends",           # 数据集名称
            data=np.array(episode_ends_arrays), # 要写入的数据（numpy array，记录每个episode的结束帧下标）
            dtype="int64",            # 数据类型为 int64
            overwrite=True,           # 如果已存在同名数据集则覆盖
            compressor=compressor,    # 使用前面定义的压缩器（zstd压缩）
        )
        
            
        
        

if __name__ == "__main__":
    main()
    