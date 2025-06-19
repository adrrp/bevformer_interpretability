import os
import shutil
import gc
import json
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from projects.mmdet3d_plugin.metrics import IntersectionOverUnion
import mmcv
from saliency_techniques.util import parse_args, init_model_data, suppress_stdout
from saliency_techniques.generate_perturbed_data import generate_perturbed_images




"""
args = parse_args()
cfg, model, dataset, data_loader = init_model_data(args)
nds, iou = model_inference(cfg, model, data_loader)
print("NDS: ", nds, " | IoU: ", iou)
"""
def model_inference(cfg, model, data_loader):
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    outputs = []
    dataset = data_loader.dataset

    num_map_class = 4
    semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            in_data = {i: j for i, j in data.items() if 'img' in i}
            result = model(return_loss=False, rescale=True, **in_data)

        batch_size = len(result)
        
        #BBOX
        outputs.extend([dict(pts_bbox=result[0]['pts_bbox'])])

        #SEGM
        pred = result[0]['seg_preds']
        max_idx = torch.argmax(pred, dim=1, keepdim=True)               # Per cada pixel pilla l'idx de la clase amb major prob
        one_hot = pred.new_full(pred.shape, 0)                          # Crea un tensor de 0s amb igual forma que pred
        one_hot.scatter_(1, max_idx, 1)                                 # dim=1 (dimensio de clase), a la dimesio de clase aplica segons max_idx el valor 1
        
        num_cls = pred.shape[1]                                                 #Numero de clases
        indices = torch.arange(0, num_cls).reshape(-1, 1, 1).to(pred.device)    # indices tensor (C,1,1) on primera dimensió te idxs de les clases (0...C-1)
        pred_semantic_indices = torch.sum(one_hot * indices, axis=1).int()      # Multiplica en eix, acaba indicant quina clase ha predit cada pixel
        target_semantic_indices = data['semantic_indices'][0].cuda()            # Etiquetes reals de segmentació
        semantic_map_iou_val(pred_semantic_indices, target_semantic_indices)    # Calcula IoU

        for _ in range(batch_size):
            prog_bar.update()


    
    scores = semantic_map_iou_val.compute()
    mIoU = sum(scores[1:]) / (len(scores) - 1)
    

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for k in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(k, None)
    
    with suppress_stdout():
        evl = dataset.evaluate(outputs, **eval_kwargs)
    
    return evl["pts_bbox_NuScenes/NDS"], mIoU.cpu().detach().numpy()


def run_perturbation_tests(
    args,
    original_dir,
    backup_dir,
    perturbed_base,
    json_output_path,
    saliency_path,
):
    cfg, model, dataset, data_loader = init_model_data(args)

    results = {
        "positive": {"nds": {}, "iou": {}},
        "negative": {"nds": {}, "iou": {}}
    }

    print("Generant dades base...")
    nds, iou = model_inference(cfg, model, data_loader)
    base_nds = round(float(nds), 4)
    base_iou = round(float(iou), 4)
    print(f"NDS: {base_nds}, IoU: {base_iou}")

    results["positive"]["nds"]["0"] = base_nds
    results["positive"]["iou"]["0"] = base_iou
    results["negative"]["nds"]["0"] = base_nds
    results["negative"]["iou"]["0"] = base_iou

    if not os.path.exists(backup_dir):
        print("Moviendo carpeta original a backup...")
        os.rename(original_dir, backup_dir)

    perc_positive = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    perc_negative = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


    def run_single_perturbation(percent, positive):
        mode = 'positive' if positive else 'negative'
        percent_str = str(int(percent * 100))

        print(f"\n=== Ejecutando {mode} perturbation con {percent_str}% ===")
        out_dir = f"{perturbed_base}_{mode}_{percent_str}"

        generate_perturbed_images(
            percent=percent,
            positive=positive,
            pkl_path='./data/nuscenes/nuscenes_infos_temporal_val.pkl',
            img_root=backup_dir,
            out_root=out_dir,
            saliency_path=saliency_path
        )

        if os.path.exists(original_dir):
            shutil.rmtree(original_dir)
        os.rename(out_dir, original_dir)

        # Inferencia
        nds, iou = model_inference(cfg, model, data_loader)
        results[mode]["nds"][percent_str] = round(float(nds), 4)
        results[mode]["iou"][percent_str] = round(float(iou), 4)
        print(f"NDS: {round(float(nds), 4)}, IoU: {round(float(iou), 4)}")

        if os.path.exists(original_dir):
            shutil.rmtree(original_dir)
        gc.collect()

    for p in perc_positive:
        run_single_perturbation(p, positive=True)

    for p in perc_negative:
        run_single_perturbation(p, positive=False)

    # Restaurar carpeta original
    if os.path.exists(original_dir):
        shutil.rmtree(original_dir)
    os.rename(backup_dir, original_dir)

    print("Results: ")
    print(results)

    with open(json_output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResultados guardados en {json_output_path}")
    return results




def main():
    args = parse_args()


    print("SALIENCY PERTURBATION TEST")
    original_dir='./data/nuscenes/samples'
    backup_dir='./data/nuscenes/samples_backup'
    perturbed_base='./data/nuscenes/samples_perturbation'
    json_output_path='./saliency_techniques/perturbation_tests/ig_bbox_base.json'
    saliency_path='./saliency_techniques/npy-maps/saliency-global-bbox-base/global_saliency.npy'

    results = run_perturbation_tests(
        args=args,
        original_dir=original_dir,
        backup_dir=backup_dir,
        perturbed_base=perturbed_base,
        json_output_path=json_output_path,
        saliency_path=saliency_path
    ) 




if __name__ == '__main__':
    main()
    