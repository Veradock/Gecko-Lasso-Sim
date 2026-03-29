import cv2
import numpy as np

def speed_up_and_crop(
    input_path,
    output_path,
    target_length_sec,
    crop_width,
    crop_height,
    crop_x=None,
    crop_y=None
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Original properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        raise ValueError("FPS is zero.")

    actual_length_sec = total_frames / fps

    if target_length_sec >= actual_length_sec:
        raise ValueError("Target length must be shorter than original.")

    print(f"Original length: {actual_length_sec:.2f} sec")

    # --- Compute exact number of output frames ---
    target_frame_count = int(round(target_length_sec * fps))
    print(f"Target frames: {target_frame_count}")

    # Evenly spaced frame indices
    indices = np.linspace(
        0,
        total_frames - 1,
        target_frame_count,
        dtype=np.int32
    )

    # --- Crop handling ---
    if crop_width > frame_width or crop_height > frame_height:
        raise ValueError("Crop exceeds original dimensions.")

    if crop_x is None:
        crop_x = (frame_width - crop_width) // 2
    if crop_y is None:
        crop_y = (frame_height - crop_height) // 2

    if (crop_x < 0 or crop_y < 0 or
            crop_x + crop_width > frame_width or
            crop_y + crop_height > frame_height):
        raise ValueError("Invalid crop region.")

    # --- Setup writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    # --- Jump directly to relevant frames ---
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        print(idx)
        cropped = frame[
            crop_y:crop_y + crop_height,
            crop_x:crop_x + crop_width
        ]

        out.write(cropped)

    cap.release()
    out.release()

    print("Done.")
    print(f"Final duration ≈ {target_frame_count / fps:.2f} sec")


if __name__ == "__main__":
    speed_up_and_crop(
        input_path="/Users/matthew/Library/CloudStorage/OneDrive-Stanford/Desktop/Trimmed (0 0p71 0p71 2 radps 5N offsetCM).mov",
        output_path="/Users/matthew/Library/CloudStorage/OneDrive-Stanford/Desktop/Exported Trimmed (0 0p71 0p71 2 radps 5N offsetCM).mp4",
        target_length_sec=21.2,
        crop_width=2375,
        crop_height=2112,
        crop_x=None,  # None = center crop
        crop_y=122
    )