import argparse
import torch
import numpy as np
from os.path import join
import os
import cv2
from tqdm import tqdm
from model import recon_model as model_arch
from data_loader.data_loaders import InferenceDataLoader
from utils.util import CropParameters, get_height_width, torch2cv2, torch2cv2_u16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tonemapping(img, adapted_lum):
    middle_grey = 1.0
    color = img * (middle_grey / adapted_lum)
    return color / (1.0 + color)

def quick_norm(img):
    return (img - torch.min(img))/(torch.max(img) - torch.min(img) + 1e-5)

def normalize_image_sequence(sequence):
    images = torch.stack([item for item in sequence], dim=0)
    mini = np.percentile(torch.flatten(images), 1)
    maxi= np.percentile(torch.flatten(images), 99)
    images = (images - mini) / (maxi - mini + 1e-5)
    images = torch.clamp(images, 0, 1)
    for i in range(len(sequence)):
        sequence[i] = images[i, ...]
    return sequence

def load_model(checkpoint):
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model

def main(args, model):
    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': None,
                      'num_bins': 5,
                      'combined_voxel_channels': False,
                      'voxel_method': {'method': args.voxel_method,
                                       'k': args.k,
                                       't': args.t,
                                       'sliding_window_w': args.sliding_window_w,
                                       'sliding_window_t': args.sliding_window_t}
                      }

    print(args.events_file_path)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type)

    height, width = get_height_width(data_loader)
    crop = CropParameters(width, height, 4)

    model.reset_states()
    recon_seq, ref_seq = [], []
    for i, item in enumerate(tqdm(data_loader)):
        voxel = item['events'].to(device)
        voxel = crop.pad(voxel)
        # For event preview
        event_prev = torch.sum(voxel, dim=2, keepdim=True)
        event_prev = quick_norm(event_prev[0, 0, ...].detach().cpu())
        with torch.no_grad():
            output = model(voxel)
        image = crop.crop(output['image'])
        recon_seq.append(image.detach().cpu())
        ref_seq.append(item['frame'].detach().cpu())
        event_cv = torch2cv2(crop.crop(event_prev))
        ename = 'events_{:010d}.png'.format(i)
        cv2.imwrite(join(args.output_folder, ename), event_cv)
    recon_seq = normalize_image_sequence(recon_seq)
    ref_seq = normalize_image_sequence(ref_seq)
    for i, image in enumerate(recon_seq):
        image_cv = torch2cv2_u16(image)
        ref_cv = torch2cv2_u16(ref_seq[i])
        fname = 'recon_{:010d}.png'.format(i)
        refname = 'ref_{:010d}.png'.format(i)
        image_cv = tonemapping(image_cv, 30000.)
        ref_cv = tonemapping(ref_cv, 30000.)
        cv2.imwrite(join(args.output_folder, fname), image_cv*255)
        cv2.imwrite(join(args.output_folder, refname), ref_cv*255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--events_file_path', required=True, type=str,
                        help='path to events (HDF5)')
    parser.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('--voxel_method', default='between_frames', type=str,
                        help='which method should be used to form the voxels',
                        choices=['between_frames', 'k_events', 't_seconds'])
    parser.add_argument('--k', type=int,
                        help='new voxels are formed every k events (required if voxel_method is k_events)')
    parser.add_argument('--sliding_window_w', type=int,
                        help='sliding_window size (required if voxel_method is k_events)')
    parser.add_argument('--t', type=float,
                        help='new voxels are formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sliding_window_t', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--loader_type', default='H5', type=str,
                        help='Which data format to load (HDF5 recommended)')


    args = parser.parse_args()
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    print('Loading checkpoint: {} ...'.format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)
    model = load_model(checkpoint)
    main(args, model)
