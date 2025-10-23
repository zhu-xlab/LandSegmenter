import os
import sys
import cv2
import torch
import rasterio
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sam2.get_args import get_args_ov as get_args
from proxyclip_segmentor import ProxyCLIPSegmentation
from matplotlib.colors import ListedColormap


colors = ['#440154', '#99FFCC', '#808080', '#E86ACD', '#FFFF19', '#008000']
cmap_potsdam = ListedColormap(colors)
colors = ['#FFFFFF', '#0000FF', '#00823C', '#9BBB59', '#99CCFF', '#FF9900', '#440154', '#C00000', '#FFFF19', '#DDDDDD']
cmap_dw = ListedColormap(colors) 
colors = ['#FFFFFF', '#FF0000', '#FFFF19', '#0000FF', '#7030A0', '#00FF00', '#E97132']
cmap_loveda = ListedColormap(colors)
colors = ['#00823C', '#9BBB59', '#FF9900', '#0000FF', '#FFFF19', '#5F5F5F', '#DDDDDD', '#C00000']
cmap_nyc = ListedColormap(colors)
colors = ['#FFFFFF', '#C00000', '#FF9900', '#00823C', '#FF00FF', '#CCFFCC', '#DDDDDD', '#9BBB59', '#B97B3D', '#0000FF', '#FFFF19', '#440154', '#99CCFF', '#5BFFFF']
cmap_osm = ListedColormap(colors)
colors = ['#FF5149', '#FF9900', '#D86DCD', '#9BBB59', '#DDDDDD', '#FFFF99', '#FFCCFF', '#9A7087', '#C1F0C8', '#00B050', '#00823C', '#B97B3D', '#99CCFF', '#0000FF']
cmap_multisenge = ListedColormap(colors)
CMAPS = {'potsdam': cmap_potsdam,
         'loveda': cmap_loveda,
         'nyc': cmap_nyc,
         'dw': cmap_dw,
         'osm': cmap_osm,
         'multisenge': cmap_multisenge,
         }
VMAXS = {'potsdam': 5,
         'loveda': 6,
         'nyc': 7,
         'dw': 9,
         'osm': 13,
         'multisenge': 13,
         }
KEY_IMG = 'image'
KEY_LBL = 'label'
KEY_DATASAT = 'dataset'
KEY_WAVE = 'waves'
KEY_WEAKLBL = 'weak'
IMG_PATH = {'potsdam': 'demo_data/image/Potsdam_5_15_1024_4096.png',
            'loveda': 'demo_data/image/LoveDA_Urban_3662_512_512.png',
            'nyc': 'demo_data/image/NYC_11_6138.png',
            'dw': 'demo_data/image/DW_85p4153719505_26p8204800842-20190327_topleft.tif',
            'osm': 'demo_data/image/OSM_31.7099137974_49.6452367157-20190907_topleft.tif',
            'multisenge': 'demo_data/image/MultiSenGe_32TLT_20200807_S2_5654_2570.tif',
            }
MEANS = {'potsdam': [123.675, 116.28, 103.53],
         'loveda': [123.675, 116.28, 103.53],
         'nyc': [111.34724539, 130.45546838, 130.80935808],
         'dw':  [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312],
         'osm': [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312],
         'multisenge':[884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2368.51236873, 1805.06846033],
         }
STDS  = {'potsdam': [58.395, 57.12, 57.375],
         'loveda': [58.395, 57.12, 57.375],
         'nyc': [46.64435224, 36.81250866, 32.99007486],
         'dw':  [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676], 
         'osm': [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676],
         'multisenge': [1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1455.52084939, 1343.48379601],
         }
WAVES = {'potsdam': [0.665, 0.56, 0.49],
         'loveda': [0.665, 0.56, 0.49],
         'nyc': [0.665, 0.56, 0.49],
         'dw':  [0.443, 0.490, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.940, 1.375, 1.61, 2.19], 
         'osm': [0.443, 0.490, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.940, 1.375, 1.61, 2.19],
         'multisenge': [0.490, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 1.61, 2.19],
         }
BANDS_ORDER = {'potsdam': [0,1,2],
               'loveda': [0,1,2],
               'nyc': [0,1,2],
               'dw':  [3,2,1,0,4,5,6,7,8,9,10,11,12], 
               'osm': [3,2,1,0,4,5,6,7,8,9,10,11,12],
               'multisenge': [2,1,0,3,4,5,6,7,8,9],
               }

def single_band_normalize(img, mean, std, clip=False):
    if clip:
        scale = 3  
        min_value = mean - scale * std
        max_value = mean + scale * std
        img = np.clip(img, min_value, max_value)
    img = (img-mean)/std
    return img

def main():
    ###### 1 - parameter setting ######
    # data selection variable
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo_dataset", dest='demo_dt', type=str, default="dw", choices=['potsdam', 'loveda', 'nyc', 'dw', 'osm', 'multisenge'],
                        help="Dataset name for one-sample demo implementation.")
    extra_args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    # other defined parameters
    auto_defaults = {"test_only": True, "accelerator": "gpu"}
    args = get_args(defaults=auto_defaults)
    args.test_only = True
    args.use_aux_decoder = True
    args.ts_wpath = 'checkpoints/LandSegmenter-aux-epoch=49.ckpt'
    args.test_use_multiple_prompts = True
    args.vfm_model = 'weak'
    args.weak_feat_type = 'encoder_embed'
    args.save_dir = 'demo_data'
    args.dofa_type = 'vit_base_patch16'
    args.dofa_ckpt_path = 'checkpoints/DOFA_ViT_base_e100.pth'
    # merge args
    args.__dict__.update(vars(extra_args))
    args.dataset_name = args.demo_dt
    print(f'Implement DEMO on the image from {args.demo_dt}')

    # ##### 2 - prepare data
    fp = IMG_PATH[args.demo_dt]
    if fp.endswith('png'):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        img_show = img[:, :, ::-1]
        img = img_show.transpose(2,0,1).astype(np.float32)
    elif fp.endswith('.tif'):
        with rasterio.open(fp) as src:
            img = src.read().astype(np.float32)
    
    # normalization
    clip_opt = (img.shape[0]!=3)
    for bi, (mean, std) in enumerate(zip(MEANS[args.demo_dt], STDS[args.demo_dt])):
        img[bi] = single_band_normalize(img[bi], mean, std, clip=clip_opt)
    
    # rearrange band order: RGB+others
    img = img[BANDS_ORDER[args.demo_dt]]
    if fp.endswith('.tif'):
        img_show = img[:3].transpose(1,2,0)
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())
        img_show = np.clip(img_show*1.5,0,1)
    x = torch.tensor(img)[None]
    waves = np.array(WAVES[args.demo_dt])[BANDS_ORDER[args.demo_dt]]

    # prepare batch
    batch = {KEY_IMG: x,
             KEY_WAVE: torch.tensor(waves)[None],
             KEY_DATASAT: [args.demo_dt],
             KEY_WEAKLBL: [False],
             }
    x = x.contiguous().to('cuda')
        
    ###### 3 - prepare model ######
    model = ProxyCLIPSegmentation(clip_type=args.clip_type, 
                                  model_type=args.clip_visual_type, 
                                  vfm_model=args.vfm_model,
                                  name_path=f'ProxyCLIP/configs/cls_{args.dataset_name}.txt',
                                  slide_stride=args.clip_slide_stride, slide_crop=args.clip_slide_crop,
                                  args=args,
                                  )
    
    ###### 4 - make predictions ######
    clip_prob = model.predict(x, batch, return_prob=True)
    weak_prob = model.get_weak_model_prediction(batch, return_prob=True)
    weak_prob_max = weak_prob.max(-1)[0].max(-1)[0][:,:,None,None]
    clip_prob_max = clip_prob.max(-1)[0].max(-1)[0][:,:,None,None]
    mask_clip = torch.ones_like(weak_prob_max)*2
    mask_weak = torch.ones_like(weak_prob_max)*2
    select = (weak_prob_max<args.max_thr)*(clip_prob_max>=args.max_thr)
    mask_clip[select] = 3
    mask_weak[select] = 1
    select = (weak_prob_max>=args.max_thr)*(clip_prob_max<args.max_thr)
    mask_clip[select] = 1
    mask_weak[select] = 3
    seg_pred = mask_clip*clip_prob+mask_weak*weak_prob
    seg_pred = seg_pred/seg_pred.sum(dim=1, keepdim=True)
    # # # # # # uneven combination
    seg_pred = seg_pred.argmax(1).squeeze().cpu().numpy()
    seg_clip = clip_prob.argmax(1).squeeze().cpu().numpy()
    seg_weak = weak_prob.argmax(1).squeeze().cpu().numpy()

    if args.demo_dt in ['dw', 'osm']:
        seg_pred = seg_pred+1
        seg_clip = seg_clip+1
        seg_weak = seg_weak+1
    
    # show and save results
    plt.figure(figsize=(16,4))
    plt.subplot(141)
    plt.imshow(img_show)
    plt.axis('off')
    plt.title('Image')
    plt.subplot(142)
    plt.imshow(seg_weak, interpolation='none', cmap=CMAPS[args.demo_dt], vmin=0, vmax=VMAXS[args.demo_dt])
    plt.axis('off')
    plt.title('LandSegmenter')    
    plt.subplot(143)
    plt.imshow(seg_clip, interpolation='none', cmap=CMAPS[args.demo_dt], vmin=0, vmax=VMAXS[args.demo_dt])
    plt.axis('off')
    plt.title('ProxyCLIP (w LandSegmenter)')
    plt.subplot(144)
    plt.imshow(seg_pred, interpolation='none', cmap=CMAPS[args.demo_dt], vmin=0, vmax=VMAXS[args.demo_dt])
    plt.axis('off')
    plt.title('Fusion')
    dir_save = 'demo_data/predict'
    if not os.path.exists(dir_save): os.mkdir(dir_save)
    fp_save = os.path.join(dir_save, f'seg_maps_{args.demo_dt}.png')
    plt.savefig(fp_save, bbox_inches='tight', pad_inches=0.3, dpi=600)

if __name__ == '__main__':
    main()