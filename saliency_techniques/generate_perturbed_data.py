import os
import pickle
import cv2
import numpy as np
import mmcv



def generate_perturbed_images(percent, positive, pkl_path, img_root, out_root, saliency_path):
    with open(pkl_path, 'rb') as f:
        infos_dict = pickle.load(f)
    infos = infos_dict['infos']

    saliencys = np.load(saliency_path, allow_pickle=True)  # (num_frames, num_cams, H, W)

    img_paths = []
    for info in infos:
        cams = info.get('cams', {})
        frame_paths = [cams[cam_name].get('data_path') for cam_name in sorted(cams.keys())]
        img_paths.append(frame_paths)

    num_frames = len(img_paths)
    num_cams = len(img_paths[0])
    print(f"Processant {num_frames} frames x {num_cams} cams")

    prog_bar = mmcv.ProgressBar(num_frames)

    for frame_idx in range(num_frames):
        for cam_idx in range(num_cams):

            rel_path = img_paths[frame_idx][cam_idx]
            rel_to_samples = os.path.relpath(rel_path, 'samples') if rel_path.startswith('samples') else os.path.relpath(rel_path, './data/nuscenes/samples')
            in_path = os.path.join(img_root, rel_to_samples)
            out_path = os.path.join(out_root, rel_to_samples)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            img = cv2.imread(in_path)
            h, w = img.shape[:2]

            sal = saliencys[frame_idx][cam_idx]
            h_s, w_s = sal.shape

            sal_norm = (sal - sal.min()) / (sal.max() - sal.min())
            flat_sal = sal_norm.flatten()
            num_pixels = flat_sal.size
            num_to_mask = int(percent * num_pixels)

            if positive:
                indices = np.argsort(-flat_sal)[:num_to_mask]
            else:
                indices = np.argsort(flat_sal)[:num_to_mask]

            flat_mask = np.zeros(num_pixels, dtype=bool)
            flat_mask[indices] = True
            mask = flat_mask.reshape(h_s, w_s)

            if mask.shape != (h, w):
                mask = mask[:h, :]

            mean_pixel = img.mean(axis=(0, 1)).astype(np.uint8)
            img[mask] = mean_pixel
            cv2.imwrite(out_path, img)

        prog_bar.update()


if __name__ == '__main__':
    PERCENT = 0.30
    POSITIVE = True
    PKL_PATH = './data/nuscenes/nuscenes_infos_temporal_val.pkl'
    IMG_ROOT = './data/nuscenes/samples/'
    OUT_ROOT = './data/nuscenes/samples_perturbation/'
    SALIENCY_PATH = './saliency_techniques/npy-maps/saliency-global-segm-base/global_saliency.npy'

    generate_perturbed_images(PERCENT, POSITIVE, PKL_PATH, IMG_ROOT, OUT_ROOT, SALIENCY_PATH)