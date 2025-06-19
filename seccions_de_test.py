
import numpy as np    

""" GradCam primigeni
    FRAME_INTERPRET = 5  # Sólo interpretamos este frame
    TARGET_CLASS = 0     # Índice de la clase que queremos visualizar

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        act_feats, grad_feats = [], []

        # 1) Hook en la última conv del seg_decoder
        target_module = model.module.pts_bbox_head.seg_decoder.up2[1]
        def forward_hook(mod, inp, out):
            # out: tensor (B, C, H_bev, W_bev)
            act_feats.append(out)
        target_module.register_forward_hook(forward_hook)

        outputs = []
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
        print(f"[INFO] Inferencia sobre {len(data_loader.dataset)} muestras…")

        for idx, sample in enumerate(data_loader):
            # 2) Inferencia normal sin grad
            with torch.no_grad():
                res = model(return_loss=False, rescale=True,
                            img=sample['img'], img_metas=sample['img_metas'])
            outputs.extend(res)
            prog_bar.update(len(res))

            # 3) Solo en el frame de interés hacemos Grad-CAM
            if idx == FRAME_INTERPRET:
                act_feats.clear()
                grad_feats.clear()
                model.zero_grad()

                # Desconectar prev_bev para romper ciclos en el grafo
                if hasattr(model.module, 'prev_bev'):
                    model.module.prev_bev = model.module.prev_bev.detach()

                # Forward con grafo activo para capturar activaciones
                out = model(return_loss=False, rescale=True,
                            img=sample['img'], img_metas=sample['img_metas'])
                # out es una lista de dicts, donde res[0]['seg_preds'] tiene forma (1, num_cls, H_bev, W_bev)
                pred = out[0]['seg_preds']       # shape: (B=1, C, H, W)
                assert pred is not None, "No se obtuvieron preds de segmentación"

                # 4) Preparamos activación A
                A = act_feats[-1]                # (1, C_dec, H_bev, W_bev)
                A.retain_grad()
                def tensor_grad_hook(grad):
                    grad_feats.append(grad)
                A.register_hook(tensor_grad_hook)

                # 5) Extraemos score de la clase objetivo del seg head
                #    Puede que quieras subir pred al mismo tamaño que A o viceversa,
                #    pero aquí asumimos que coinciden H,W de pred y A.
                score_map = pred[:, TARGET_CLASS, :, :]      # (1, H_bev, W_bev)
                score = score_map.mean()                     # escalar agregando sobre mapa
                score.backward(retain_graph=True)

                # 6) Construcción del CAM
                G = grad_feats[0]        # (1, C_dec, H_bev, W_bev)
                B, C, H, W = A.shape
                weights = G[0].mean(dim=(1,2))              # α_c para cada canal
                cam = (weights.view(C,1,1) * A[0]).sum(dim=0)
                cam = cam.clamp(min=0)
                cam = cam.detach().cpu().numpy()
                cam = (cam - cam.min())/(cam.max()-cam.min()+1e-6)

                # 7) Superposición sobre imágenes originales (igual que antes)
                dc = sample['img'][0]; data = dc.data
                imgs = data[0] if isinstance(data, (list, tuple)) else data
                if imgs.dim() == 5:
                    imgs = imgs.squeeze(0)

                for cam_id in range(imgs.shape[0]):
                    img = imgs[cam_id].cpu().permute(1,2,0).numpy()
                    img = (img - img.min())/(img.max()-img.min()+1e-6)
                    img = (img*255).astype(np.uint8)

                    # colorear CAM y resize
                    cam_u8   = (cam * 255).astype(np.uint8)
                    heatmap  = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
                    h_img, w_img = img.shape[:2]
                    heatmap = cv2.resize(heatmap, (w_img, h_img))

                    overlay = cv2.addWeighted(img[...,::-1], 0.6, heatmap, 0.4, 0)
                    out_path = f'./outputs/gradcam/gradcam_class{TARGET_CLASS}_frame{idx}_cam{cam_id}.png'
                    cv2.imwrite(out_path, overlay)

                print(f"[Grad-CAM] Visualizaciones completadas para clase {TARGET_CLASS} en frame {idx}.\n")

        print("\n[INFO] Inferencia y Grad-CAM completados.")
    """



    """ Traduccio single gpu test
    map_enable = True
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        outputs = []
        dataset = data_loader.dataset
        
        bev_feats = []
        def grab_bev(module, inp, outp):
            bev_feats.append(outp.detach().cpu())
        enc = model.module.pts_bbox_head.transformer.encoder
        enc.register_forward_hook(grab_bev)

        if map_enable:
            num_map_class = 4
            semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()

        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):

            with torch.no_grad():
                in_data = {i: j for i, j in data.items() if 'img' in i}
                result = model(return_loss=False, rescale=True, **in_data)

            batch_size = len(result)
            
            
            if result[0]['pts_bbox'] != None:
                outputs.extend([dict(pts_bbox=result[0]['pts_bbox'])])

            if result[0]['seg_preds'] is not None:
                pred = result[0]['seg_preds']
                #Hacemos one hot encoding
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



        if map_enable:
            import prettytable as pt
            scores = semantic_map_iou_val.compute()
            mIoU = sum(scores[1:]) / (len(scores) - 1)
            tb = pt.PrettyTable()
            tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
            tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                        round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                        round(mIoU.cpu().numpy().item(), 4)])
            print('\n')
            print(tb)

        if args.save_raw_bev:
            DATAROOT = './data/nuscenes'
            nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
            os.makedirs(args.save_raw_bev, exist_ok=True)
            ds = dataset

            for idx, bev in enumerate(bev_feats):
                info = ds.data_infos[idx]
                scene = nusc.get('scene', info['scene_token'])
                log   = nusc.get('log', scene['log_token'])
                scene_name = log['logfile']

                scene_dir = osp.join(args.save_raw_bev, scene_name)
                os.makedirs(scene_dir, exist_ok=True)

                prefix = scene_name[:5]
                fname = f'{prefix}{idx:04d}.npy'
                path = osp.join(scene_dir, fname)
                np.save(path, bev.numpy())

            print(f"[INFO] Saved {len(bev_feats)} raw BEV features to {args.save_raw_bev}")
    """


    """ Salience ma o meno
    model = MMDataParallel(model, device_ids=[0])
    print(f"[DEBUG] Model wrapped in MMDataParallel; device_ids = {model.device_ids}")

    print("[DEBUG] Detector listo, pasando a modo eval()")
    model.eval()

    _device = next(model.module.parameters()).device
    print(f"[DEBUG] Parámetro de modelo en device: {_device}")

    FRAME_INTERPRET = 3
    TARGET_CLASS   = 0

    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))

    for idx, sample in enumerate(data_loader):
        print(f"\n[DEBUG] Batch idx = {idx}")

        # 3.1) Forward normal sin gradientes
        with torch.no_grad():
            res = model(
                return_loss=False,
                rescale=True,
                img=sample['img'],
                img_metas=sample['img_metas']
            )
        print(f"[DEBUG]   → Respuestas obtenidas: {len(res)} items")
        prog_bar.update(len(res))

        # 3.2) Solo en el batch de interés: saliency
        if idx == FRAME_INTERPRET:
            print("[DEBUG] ¡Frame de interés! Extrayendo tensor de imagen…")
            img_dc = sample['img'][0]          # DataContainer
            print(f"[DEBUG]   DataContainer contiene {len(img_dc.data)} elementos")
            tensor = img_dc.data[0]            # Tensor (num_cams,3,H,W)
            print(f"[DEBUG]   Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

            imgs = (tensor.clone()
                        .detach()
                        .requires_grad_(True)
                        .to(_device))
            print(f"[DEBUG]   imgs.requires_grad = {imgs.requires_grad}; device = {imgs.device}")

            # Metadatos
            meta_dc = sample['img_metas'][0]
            metas   = meta_dc.data[0]
            print(f"[DEBUG]   Metadatos extraídos: {type(metas)}")

            # 3.3) Forward con gradientes ACTIVOS
            print("[DEBUG] Forward_test con torch.enable_grad()…")
            with torch.enable_grad():
                out = model.module.forward_test(
                    img=[imgs],
                    img_metas=[metas]
                )
            print(f"[DEBUG]   forward_test returned keys: {list(out[0].keys())}")

            seg_preds = out[0].get('seg_preds', None)
            print(f"[DEBUG]   seg_preds shape: {None if seg_preds is None else seg_preds.shape}")
            if seg_preds is None:
                raise RuntimeError("No se generó 'seg_preds' en el output.")

            # 3.4) Score y backward
            score = seg_preds[:, TARGET_CLASS].mean()
            print(f"[DEBUG]   Score para clase {TARGET_CLASS}: {score.item():.6f}")
            model.zero_grad()
            score.backward(retain_graph=False)

            # 3.5) Extraer gradientes
            grads = imgs.grad
            print(f"[DEBUG]   imgs.grad es None? {grads is None}")
            if grads is None:
                raise RuntimeError("¡gradiente nulo! Algo impidió el backward.")
            print(f"[DEBUG]   grads.shape: {grads.shape}")

            cams = []
            for v in range(grads.size(0)):
                sal = torch.norm(grads[v], dim=0).cpu().numpy()
                sal = (sal - sal.min())/(sal.max()-sal.min()+1e-6)
                cams.append(sal)
            print(f"[DEBUG]   Computadas {len(cams)} saliency maps")

            # 3.6) Superponer y guardar
            origs = imgs.detach().cpu().permute(0,2,3,1).numpy()
            for v, (img, cam) in enumerate(zip(origs, cams)):
                print(f"[DEBUG]     Procesando cámara {v} — img shape {img.shape}, cam shape {cam.shape}")
                img_u8 = np.uint8(255 * np.clip(img,0,1))
                cam_u8 = np.uint8(255 * cam)
                heat   = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_u8[...,::-1], 0.6, heat, 0.4, 0)
                out_path = f'./outputs/saliency/cam{v}_cls{TARGET_CLASS}.png'
                cv2.imwrite(out_path, overlay)
                print(f"[DEBUG]       Guardado → {out_path}")

            print(f"[Saliency] mapas guardados para clase {TARGET_CLASS}, frame {idx}")
            break

    print("Terminado.")
    """

    # Distributed launch
    """
    if distributed:
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        
        map_enable = True
        model.eval()
        outputs = []
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()

        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))

        if map_enable:
            num_map_class = 4
            semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()

        time.sleep(2)  # Evita deadlocks

        for i, data in enumerate(data_loader):
            with torch.no_grad():
                in_data = {i: j for i, j in data.items() if 'img' in i}
                result = model(return_loss=False, rescale=True, **in_data)

                batch_size = len(result)

                if result[0]['pts_bbox'] is not None:
                    outputs.extend([dict(pts_bbox=result[0]['pts_bbox'])])

                if result[0]['seg_preds'] is None:
                    map_enable = False

                if map_enable:
                    pred = result[0]['seg_preds']
                    max_idx = torch.argmax(pred, dim=1, keepdim=True)
                    one_hot = pred.new_full(pred.shape, 0)
                    one_hot.scatter_(1, max_idx, 1)

                    num_cls = pred.shape[1]
                    indices = torch.arange(0, num_cls).reshape(-1, 1, 1).to(pred.device)
                    pred_semantic_indices = torch.sum(one_hot * indices, axis=1).int()
                    target_semantic_indices = data['semantic_indices'][0].cuda()
                    semantic_map_iou_val(pred_semantic_indices, target_semantic_indices)

            if rank == 0:
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        print("Preprintint results")
        if map_enable and rank == 0:
            import prettytable as pt
            scores = semantic_map_iou_val.compute()
            mIoU = sum(scores[1:]) / (len(scores) - 1)
            tb = pt.PrettyTable()
            tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
            tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                        round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                        round(mIoU.cpu().numpy().item(), 4)])
            print('\n')
            print(tb)

        if map_enable:
            import prettytable as pt
            scores = semantic_map_iou_val.compute()
            mIoU = sum(scores[1:]) / (len(scores) - 1)
            if rank == 0:
                tb = pt.PrettyTable()
                tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
                tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                            round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                            round(mIoU.cpu().numpy().item(), 4)])
                print('\n')
                print(tb)

                #seg_dict = dict(
                #    Validation_num=len(dataset),
                #    Divider=round(scores[1:].cpu().numpy()[0], 4),
                #    Pred_Crossing=round(scores[1:].cpu().numpy()[1], 4),
                #    Boundary=round(scores[1:].cpu().numpy()[2], 4),
                #    mIoU=round(mIoU.cpu().numpy().item(), 4)
                #)
                #
                #with open('segmentation_result.json', 'a') as f:
                #    f.write(json.dumps(str(seg_dict)) + '\n')

        # Recolección de resultados
        from projects.mmdet3d_plugin.apis.test import collect_results_cpu, collect_results_gpu
        if args.gpu_collect:
            outputs = collect_results_gpu(outputs, len(dataset))
        else:
            outputs = collect_results_cpu(outputs, len(dataset), args.tmpdir)
    
    """

    """ #Intent de saliency multi-gpu, no funciona pq nomes em serveix x augmentar velocitat, no reduir vram individuals
    model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    model.eval()

    # Define la función forward para la segmentación
    def forward_fn_seg(inputs, img_meta, target_class):
        bev_feat, result = model.module.simple_test(
            img_metas=[img_meta],
            img=inputs,
            prev_bev=None,
            rescale=True
        )
        pred = result[0]['seg_preds']
        class_scores = pred[:, target_class, :, :]
        return class_scores.sum().unsqueeze(0)

    rank, _ = get_dist_info()

    # Configura la carpeta de salida para cada proceso
    root_out = f"outputs/saliency_rank{rank}"
    os.makedirs(root_out, exist_ok=True)

    # Inicializa Saliency
    saliency = Saliency(forward_fn_seg)

    # Procesa los datos
    for i, data in enumerate(data_loader):
        raw_metas = data['img_metas'][0].data[0]
        img_meta = raw_metas[0]

        cams = data['img'][0].data[0]
        inputs = cams.cuda()

        target_class = 1  # 1 = carril, 2 = paso peatones, 3 = borde

        attributions = saliency.attribute(
            inputs,
            additional_forward_args=(img_meta, target_class,)
        )  # (1,6,3,H,W)

        overlays = [None] * 6

        for cam_id in range(inputs.shape[1]):
            sal_map = attributions[0, cam_id].mean(0).cpu().detach().numpy()
            sal_map = sal_map - sal_map.min()
            if sal_map.max() > 0:
                sal_map = sal_map / sal_map.max()

            img_tensor = cams[0, cam_id]
            img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
            img_np = ((img_np - img_np.min()) /
                    (img_np.max() - img_np.min() + 1e-6) * 255).astype(np.uint8)

            H_img, W_img = img_np.shape[:2]

            sal_resized = cv2.resize(sal_map, (W_img, H_img))

            heatmap = (sal_resized * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            mask = sal_resized > 0.01
            mask_3ch = np.stack([mask]*3, axis=-1)

            overlay = img_bgr.copy()
            overlay[mask_3ch] = (img_bgr[mask_3ch] * 0.6 + heatmap[mask_3ch] * 0.4).astype(np.uint8)

            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlays[cam_id] = overlay_rgb

        fila1 = np.concatenate([overlays[2], overlays[0], overlays[1]], axis=1)
        fila2 = np.concatenate([overlays[4], overlays[3], overlays[5]], axis=1)
        final_image = np.concatenate([fila1, fila2], axis=0)

        out_path = os.path.join(root_out, f"frame_{i:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    """



  #Saliency Single GPU!!
    """
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    outputs = []
    dataset = data_loader.dataset

    #Bbox and target class = 8 ==> Pedestrian
    saliency_type = 'bbox'  # Cambiar a 'seg' o 'bbox'
    target_class = 0
    # En seg: 1 = carril, 2 = pas peatons, 3 = borde
    # En det: ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    import gc

    num_map_class = 4
    semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()

    class StatefulForwardFn:
        def __init__(self, model, saliency_type):
            self.model = model
            self.saliency_type = saliency_type
            self.prev_frame_info = {
                'prev_bev': None,
                'scene_token': None,
                'prev_pos': None,
                'prev_angle': None,
            }
            self.outputs = []

        def __call__(self, inputs, img_meta, target_class):
            # Reset prev_bev si ha cambiado de escena
            if img_meta['scene_token'] != self.prev_frame_info['scene_token']:
                self.prev_frame_info['prev_bev'] = None

            # Delta ego motion
            curr_pos = np.array(img_meta['can_bus'][:3])
            curr_angle = img_meta['can_bus'][-1]

            if self.prev_frame_info['prev_bev'] is not None:
                img_meta['can_bus'][:3] = curr_pos - self.prev_frame_info['prev_pos']
                img_meta['can_bus'][-1] = curr_angle - self.prev_frame_info['prev_angle']
            else:
                img_meta['can_bus'][:3] = 0
                img_meta['can_bus'][-1] = 0

            # Llamada al modelo
            bev_feat, result = self.model.module.simple_test(
                img_metas=[img_meta],
                img=inputs,
                prev_bev=self.prev_frame_info['prev_bev'],
                rescale=True
            )

            # Actualizar estado para siguiente frame
            self.prev_frame_info['scene_token'] = img_meta['scene_token']
            self.prev_frame_info['prev_pos'] = curr_pos
            self.prev_frame_info['prev_angle'] = curr_angle
            self.prev_frame_info['prev_bev'] = bev_feat.detach() #Fem detach per no guardar informació d'iteracions anteriors no necesaries

            # Métrica segmentación (si existe)
            if result[0]['seg_preds'] is not None:
                pred = result[0]['seg_preds']
                max_idx = torch.argmax(pred, dim=1, keepdim=True)
                one_hot = pred.new_full(pred.shape, 0)
                one_hot.scatter_(1, max_idx, 1)

                indices = torch.arange(0, pred.shape[1]).reshape(-1, 1, 1).to(pred.device)
                pred_semantic_indices = torch.sum(one_hot * indices, axis=1).int()
                target_semantic_indices = data['semantic_indices'][0].cuda()
                semantic_map_iou_val(pred_semantic_indices, target_semantic_indices)

            ## Registro de bbox output, incrementa moltisim l'ús de vram si es descomenta, no sembla existir solució senzilla al problema.
            #if result[0]['pts_bbox'] is not None:
            #    self.outputs.append(dict(pts_bbox=result[0]['pts_bbox']))

            #Registre de output, es necesari un detach si o si o la vram explota
            if result[0]['pts_bbox'] is not None:
                raw = result[0]['pts_bbox']
                boxes_3d = raw['boxes_3d'].to('cpu')
                boxes_3d.tensor = boxes_3d.tensor.detach()  # detach solo al tensor, pero sin perder clase
                cpu_result = {
                    'boxes_3d': boxes_3d,
                    'labels_3d': raw['labels_3d'].cpu().detach(),
                    'scores_3d': raw['scores_3d'].cpu().detach(),
                }
                self.outputs.append({'pts_bbox': cpu_result})

            # Escoge cómo calcular el score
            if self.saliency_type == 'seg':
                seg_preds = result[0]['seg_preds']
                if seg_preds is None:
                    return torch.tensor([0.], device=inputs.device)
                class_scores = seg_preds[0, target_class].sum()
                return class_scores.unsqueeze(0)

            elif self.saliency_type == 'bbox':
                pred = result[0]['pts_bbox']
                labels = pred['labels_3d']
                scores = pred['scores_3d']
                mask = labels == target_class
                class_scores = scores[mask]
                return class_scores.sum().unsqueeze(0)

            else:
                raise ValueError(f"saliency_type '{self.saliency_type}' no soportado")

    
    root_out = "outputs/saliency"
    os.makedirs(root_out, exist_ok=True)

    forward_fn = StatefulForwardFn(model, saliency_type)
    saliency = Saliency(forward_fn)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):  #Donde las keys de data son: img_metas, img y semantic_indices

        raw_metas = data['img_metas'][0].data[0]
        img_meta  = raw_metas[0]

        cams = data['img'][0].data[0]          # Tensor [1,6,3,H,W]
        inputs = cams.cuda()

        attributions = saliency.attribute(
            inputs,
            additional_forward_args=(img_meta,target_class,)
        ) # (1,6,3,H,W)

        torch.cuda.empty_cache()
        gc.collect()

        overlays = [None] * 6  # Lista para guardar imágenes en orden de cam_id

        #Genera imagenes de saliency
        for cam_id in range(inputs.shape[1]):
            # 1) Obtén la saliencia y normalízala
            sal_map = attributions[0, cam_id].mean(0).cpu().detach().numpy()
            sal_map = sal_map - sal_map.min()
            if sal_map.max() > 0:
                sal_map = sal_map / sal_map.max()

            # 2) Recupera la imagen original de la cámara
            img_tensor = cams[0, cam_id]
            img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
            img_np = ((img_np - img_np.min()) /
                    (img_np.max() - img_np.min() + 1e-6) * 255).astype(np.uint8)

            H_img, W_img = img_np.shape[:2]

            # 3) Redimensiona el mapa de saliencia
            sal_resized = cv2.resize(sal_map, (W_img, H_img))

            # 4) Aplica colormap
            heatmap = (sal_resized * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 5) Crear máscara binaria donde la saliencia sea mayor a un umbral pequeño
            mask = sal_resized > 0.01 # Ajusta el umbral si es necesario
            mask_3ch = np.stack([mask]*3, axis=-1)  # Expande a 3 canales

            # 6) Mezclar solo donde hay saliencia
            overlay = img_bgr.copy()
            overlay[mask_3ch] = (img_bgr[mask_3ch] * 0.6 + heatmap[mask_3ch] * 0.4).astype(np.uint8)

            # 7) Guarda la imagen resultante en RGB
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlays[cam_id] = overlay_rgb

        # Organiza las imágenes según la disposición requerida:
        fila1 = np.concatenate([overlays[2], overlays[0], overlays[1]], axis=1)
        fila2 = np.concatenate([overlays[4], overlays[3], overlays[5]], axis=1)
        final_image = np.concatenate([fila1, fila2], axis=0)  # (2H, 3W, 3)

        # Guarda la imagen final
        out_path = os.path.join(root_out, f"frame_{i:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

        prog_bar.update()
    
    outputs = forward_fn.outputs

    import prettytable as pt
    scores = semantic_map_iou_val.compute()
    mIoU = sum(scores[1:]) / (len(scores) - 1)
    tb = pt.PrettyTable()
    tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
    tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                round(mIoU.cpu().numpy().item(), 4)])
    print('\n')
    print(tb)
    
    """
    