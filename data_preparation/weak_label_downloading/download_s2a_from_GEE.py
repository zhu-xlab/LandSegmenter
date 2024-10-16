import os
import ee
import json
import pandas as pd
import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
import multiprocessing

BANDS_2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
H, W = 264, 264


def separate_df(df, n_process):
    '''
    separate df into sub dfs according to the number of processes for parallel processing
    '''
    N = df.shape[0]
    n = np.ceil(N/n_process).astype(int)

    sub_dataframes = []
    for i in range(n_process):
        start = i*n
        end = (i+1)*n
        if i==n_process-1:
            end = N
        sub_dataframes.append(df.iloc[start:end])
    return sub_dataframes


def get_s2a_img_with_id_and_bbox(img_id, coords, crs=4326):
    '''
    clip s2 images on ee using bounding box and crs
    specifically, to further avoid co-registration problem, we double check the crs of fetched images
    only fetch from the images with the same crs as required
    in this case, the fetched images should be of exactly the same size as the target/DW-label images
    '''
    region = ee.Geometry.Polygon([[coords[0][0], coords[1][1]], 
                                  [coords[1][0], coords[1][1]],
                                  [coords[1][0], coords[0][1]],
                                  [coords[0][0], coords[0][1]],
                                  [coords[0][0], coords[1][1]]]) 
    
    # gee images
    s2_sr_image = ee.Image(f'COPERNICUS/S2_SR_HARMONIZED/{img_id}')
    s2_sr = s2_sr_image.select(BANDS_2A)
    s2_sr = s2_sr.toFloat().resample('bilinear').reproject(
        s2_sr.select('B2').projection()).reproject(f'EPSG:{crs}', scale=10)
    s2_sr_img = s2_sr.toArray().sampleRectangle(region)
    image_sr = np.array(s2_sr_img.getInfo()['properties']['array'])
    
    return image_sr


def save_geotiff(img, coords, out_path):
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(coords[0][0] - xres / 2, coords[0][1] + yres / 2) * Affine.scale(xres, -yres)
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': channels,
        'crs': '+proj=latlong',
        'transform': transform,
        'dtype': img.dtype,
        'compress': 'None'
    }
    
    with rasterio.open(out_path,'w', **profile) as dst:
        for i in range(img.shape[2]):
            dst.write(img[:,:,i], i+1)
    return


def worker(args):
    ee.Initialize()
    df, save_dir = args
    n_sam = df.shape[0]
    err_meg, success_meg = [], []
    for i in range(n_sam):
        sample_id = df.iloc[i, 0]
        img_id = df.iloc[i, 1]
        coords = json.loads(df.iloc[i, 2])
        try:
            img = get_s2a_img_with_id_and_bbox(img_id, coords, crs=4326)
            assert img.shape[0] == H and img.shape[1] == W, \
                f'Image shape is not correct: {img.shape}'
            fp_save_dir = os.path.join(save_dir, f"{sample_id:07d}", "s2a")
            os.makedirs(fp_save_dir, exist_ok=True)
            fp_save = os.path.join(fp_save_dir, f'{img_id}.tif')
            save_geotiff(img, coords, fp_save)
            success_meg.append([sample_id, img_id])
        except Exception as e:
            err_meg.append([sample_id, img_id, str(e)])
    
    return {'err_meg': err_meg, 'success_meg': success_meg}


if __name__ == '__main__':
    dataset = 'worldcover_v200'
    n_process = 60
    save_dir = f'/p/scratch/hai_ws4eo/liu_ch/data/WeakFT/{dataset}'

    fp_csv = f'/p/scratch/hai_ws4eo/liu_ch/data/WeakFT/{dataset}/Downloaded_file_info_s2.csv'
    df = pd.read_csv(fp_csv, header=0)
    print(f'Total {df.shape[0]} items to download!')

    n_CPU = os.cpu_count()
    num_workers = min(n_CPU, n_process)
    err_meg, success_meg = [], []
    if num_workers>1:
        # Create a pool of worker processes
        print(f'{num_workers} cores out of {n_CPU} are used!')

        # Separate the dataframe into num_workers parts
        sub_dfs = separate_df(df, num_workers)
        data = [(sub_dfs[i], save_dir) for i in range(num_workers)]

        with multiprocessing.Pool(processes=num_workers) as p:
            results = p.map(worker, data)
        
        for r in results:
            err_meg += r['err_meg']
            success_meg += r['success_meg']
    else:
        print('No parallel processes are triggered!')
        err_meg, success_meg = worker((df, save_dir))

    # Save the error and success messages
    fp_err = os.path.join(save_dir, 'error_message_s2a.csv')
    fp_success = os.path.join(save_dir, 'success_message_s2a.csv')
    if len(err_meg)>0:
        df_err = pd.DataFrame(err_meg, columns=['sample_id', 'img_id', 'error'])
        df_err.to_csv(fp_err, index=False)
        print(f'Error messages are saved at {fp_err} ({len(err_meg)} items)')
    if len(success_meg)>0:
        df_success = pd.DataFrame(success_meg, columns=['sample_id', 'img_id'])
        df_success.to_csv(fp_success, index=False)
        print(f'Success messages are saved at {fp_success} ({len(success_meg)} items)')


