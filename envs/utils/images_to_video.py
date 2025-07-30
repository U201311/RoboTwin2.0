import cv2
import numpy as np
import os
import subprocess
import pickle
import pdb


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True) -> None:
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"
    # 尝试使用 ffmpeg 保存视频
    success = False
    
    # 首先尝试使用 libx264 编码器
    try:
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                pixel_format,
                "-video_size",
                f"{W}x{H}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                f"{out_path}",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        ffmpeg.stdin.write(imgs.tobytes())
        ffmpeg.stdin.close()
        
        if ffmpeg.wait() == 0:
            success = True
        else:
            print("libx264 encoding failed, trying alternative...")
            
    except (BrokenPipeError, OSError) as e:
        print(f"ffmpeg libx264 failed: {e}, trying alternative...")
        if 'ffmpeg' in locals():
            ffmpeg.terminate()
    
    # 如果第一次尝试失败，使用更兼容的编码器
    if not success:
        try:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    pixel_format,
                    "-video_size",
                    f"{W}x{H}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "mpeg4",
                    "-q:v",
                    "5",
                    f"{out_path}",
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            ffmpeg.stdin.write(imgs.tobytes())
            ffmpeg.stdin.close()
            
            if ffmpeg.wait() == 0:
                success = True
            else:
                print("mpeg4 encoding failed, using OpenCV...")
                
        except (BrokenPipeError, OSError) as e:
            print(f"ffmpeg mpeg4 failed: {e}, using OpenCV...")
            if 'ffmpeg' in locals():
                ffmpeg.terminate()
    
    # 如果 ffmpeg 完全失败，使用 OpenCV 作为备选方案
    if not success:
        try:
            save_video_with_opencv(imgs, out_path, fps, is_rgb)
            success = True
        except Exception as e:
            print(f"OpenCV also failed: {e}, saving as image sequence...")
            # 最终备选方案：保存为图像序列
            image_dir = out_path.replace('.mp4', '_frames').replace('.avi', '_frames')
            save_frames_as_images(imgs, image_dir, is_rgb)
            return
    
    if not success:
        raise IOError(f"All video encoding methods failed. Please check ffmpeg installation and codec support.")

    print(
        f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
    )


def save_video_with_opencv(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True) -> None:
    """使用OpenCV保存视频作为ffmpeg的备选方案"""
    n_frames, H, W, C = imgs.shape
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 定义编码器 - 使用更兼容的格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    
    if not out.isOpened():
        raise Exception("Could not open VideoWriter")
    
    for i in range(n_frames):
        frame = imgs[i]
        if is_rgb and C == 3:
            # 将RGB转换为BGR（OpenCV使用BGR）
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif C == 4:  # RGBA
            if is_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # 确保数据类型正确
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        out.write(frame)
    
    out.release()
    print(f"🎬 Video saved with OpenCV to `{out_path}`, containing {n_frames} frames at {W}×{H} resolution and {fps} FPS.")


def save_frames_as_images(imgs: np.ndarray, out_dir: str, is_rgb: bool = True) -> None:
    """将帧保存为单独的图像文件作为最后备选方案"""
    n_frames, H, W, C = imgs.shape
    
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(n_frames):
        frame = imgs[i]
        if is_rgb and C == 3:
            # 将RGB转换为BGR（OpenCV使用BGR）
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif C == 4:  # RGBA
            if is_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # 确保数据类型正确
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        frame_path = os.path.join(out_dir, f'frame_{i:06d}.png')
        cv2.imwrite(frame_path, frame)
    
    print(f"📷 Frames saved as images to `{out_dir}`, containing {n_frames} frames at {W}×{H} resolution.")
