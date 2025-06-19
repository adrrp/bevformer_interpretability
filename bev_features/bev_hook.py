import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_bev(bev_path):
    bev = np.load(bev_path)
    # Soporta shapes (B, N, C) o (N, C)
    if bev.ndim == 3:
        bev = bev[0]
    elif bev.ndim != 2:
        raise ValueError(f"Forma inesperada de BEV: {bev.shape}")
    return bev  # (N, C)


def compute_heatmap(bev, bev_h, bev_w, channel=None):
    """
    Convierte características BEV (N, C) en mapa 2D de tamaño (bev_h, bev_w).
    Si channel es None: media absoluta sobre canales.
    Si channel es int: extrae ese canal.
    """
    N, C = bev.shape
    if N != bev_h * bev_w:
        raise ValueError(f"bev_h*bev_w debe ser {bev_h*bev_w}, pero N={N})")

    if channel is None:
        vals = np.mean(np.abs(bev), axis=1)
    elif channel == -1:
        return None  # Señal especial para procesar todos los canales
    else:
        if channel < 0 or channel >= C:
            raise ValueError(f"Canal fuera de rango [0, {C-1}]")
        vals = bev[:, channel]
    heatmap = vals.reshape(bev_h, bev_w)
    return heatmap


def find_global_limits(bev_paths, bev_h, bev_w, channel=None):
    """Encuentra los valores mínimo y máximo entre todos los archivos y canales.
       Si channel es None, se usa la media absoluta sobre canales."""
    global_min = float('inf')
    global_max = float('-inf')
    
    for bev_path in bev_paths:
        bev = load_bev(bev_path)
        if channel is None:
            # Calcular los valores usando la media absoluta sobre canales
            vals = np.mean(np.abs(bev), axis=1)
            local_min = vals.min()
            local_max = vals.max()
        else:
            local_min = bev.min()
            local_max = bev.max()
        global_min = min(global_min, local_min)
        global_max = max(global_max, local_max)
    
    print(f"Rango global de valores: [{global_min:.3f}, {global_max:.3f}]")
    return global_min, global_max


def visualize_heatmap(heatmap, title=None, output=None, vmin=None, vmax=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
    plt.axis('off')
    if output:
        plt.savefig(output, bbox_inches='tight', pad_inches=0)
        print(f"Guardado mapa en: {output}")
    plt.close()


def process_single_bev(bev_path, base_output_dir, channel, bev_h, bev_w, is_from_dir=False, vmin=None, vmax=None):
    bev = load_bev(bev_path)
    if channel == -1:
        # Crear subdirectorio con el nombre del archivo bev
        bev_name = os.path.splitext(os.path.basename(bev_path))[0]
        output_subdir = os.path.join(base_output_dir, bev_name)
        os.makedirs(output_subdir, exist_ok=True)
        # Procesar cada canal
        for c in range(bev.shape[1]):
            heatmap = compute_heatmap(bev, bev_h, bev_w, channel=c)
            out_path = os.path.join(output_subdir, f'channel_{c:03d}.png')
            visualize_heatmap(heatmap, title=f'Canal {c}', output=out_path, vmin=vmin, vmax=vmax)
    else:
        heatmap = compute_heatmap(bev, bev_h, bev_w, channel)
        if is_from_dir:
            out_name = os.path.basename(bev_path).replace('.npy', '.png')
            out_path = os.path.join(base_output_dir, out_name)
        else:
            out_path = base_output_dir
        visualize_heatmap(heatmap, title=os.path.basename(bev_path), output=out_path, vmin=vmin, vmax=vmax)


def main():
    parser = argparse.ArgumentParser(
        description='Visualiza mapas BEV desde .npy')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bev-path', help='ruta al archivo .npy con BEV')
    group.add_argument('--bev-dir', help='directorio con archivos .npy')
    parser.add_argument('--channel', type=int, default=None,
                        help='número de canal a visualizar (default media, -1 para todos)')
    parser.add_argument('--output', help='ruta para guardar imagen (solo con --bev-path y channel != -1)')
    parser.add_argument('--output-dir', help='directorio para guardar pngs')
    args = parser.parse_args()

    if args.channel == -1 and not args.output_dir:
        parser.error('--output-dir es requerido cuando channel es -1')

    bev_h, bev_w = 150, 150

    if args.bev_dir:
        # Obtener lista de archivos .npy
        bev_files = [os.path.join(args.bev_dir, f) for f in sorted(os.listdir(args.bev_dir)) 
                     if f.endswith('.npy')]
        # Calcular límites globales usando args.channel
        vmin, vmax = find_global_limits(bev_files, bev_h, bev_w, channel=args.channel)
        
        os.makedirs(args.output_dir, exist_ok=True)
        for bev_file in bev_files:
            process_single_bev(bev_file, args.output_dir, args.channel, bev_h, bev_w, True, vmin, vmax)
    else:
        if args.channel != -1 and not (args.output or args.output_dir):
            parser.error('Se requiere --output u --output-dir')
        output_path = args.output_dir or args.output
        os.makedirs(os.path.dirname(output_path) if args.output else output_path, exist_ok=True)
        
        # Para un solo archivo, calcular sus propios límites usando args.channel
        vmin, vmax = find_global_limits([args.bev_path], bev_h, bev_w, channel=args.channel)
        process_single_bev(args.bev_path, output_path, args.channel, bev_h, bev_w, False, vmin, vmax)


if __name__ == '__main__':
    main()