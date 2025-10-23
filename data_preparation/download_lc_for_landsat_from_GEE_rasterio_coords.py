import os
import ee
import pandas as pd
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
import multiprocessing
from utils import read_coords, save_geotiff, separate_list, separate_df, crop_image_with_rasterio

BANDS_2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
CROP_SIZE = 264
SCALE = 30


def get_nlcd_lc_masks_from_ee(coords, fpdir, image_name, date=2019):  # or layer = landcover
    if type(date) != int: date = int(date)
    # image
    sensor = image_name.split('_')[0][2:]
    toa_image = ee.Image(f'LANDSAT/LC{sensor}/C02/T1_TOA/{image_name}')

    # collection
    collect = ee.ImageCollection(f'USGS/NLCD_RELEASES/2019_REL/NLCD').filter(ee.Filter.eq('system:index', str(date))).first()
    # if collect.projection().crs().getInfo() != 'EPSG:4326':
    collect = collect.reproject(toa_image.select('B2').projection()).reproject('EPSG:4326', scale=SCALE)

    # get data
    n_nodatas = []
    for layer in ['landcover', 'impervious']:
        select_layer = 'impervious_descriptor' if layer == 'impervious' else layer
        
        # get landcover and landuse masks
        lc = collect.select(select_layer)

        # crop image
        region = ee.Geometry.Polygon([[coords[0][0], coords[1][1]], 
                                      [coords[1][0], coords[1][1]],
                                      [coords[1][0], coords[0][1]],
                                      [coords[0][0], coords[0][1]],
                                      [coords[0][0], coords[1][1]]]) 
        
        feature = lc.toArray().sampleRectangle(region, defaultValue=0)
        image = np.array(feature.getInfo()['properties']['array'])
        org_shapes = image.shape[:2]
        if image.shape[0] != CROP_SIZE or image.shape[1] != CROP_SIZE:
            image = cv2.resize(image.astype(np.uint8), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
            image = image[:,:,np.newaxis]
        n_nodata = np.sum(image==0)
        n_nodatas.append(n_nodata)

        # save tif
        tif_path_lc = os.path.join(fpdir, f'nlcd_{layer}_{date}.tif')
        save_geotiff(image.astype(np.uint8), coords, tif_path_lc)
    return True, org_shapes, n_nodatas[0], n_nodatas[1]


def get_usfs_lc_masks_from_ee(coords, fpdir, image_name, date=2019):   # or layer = Land_Use
    if type(date) != int: date = int(date)
    # image
    sensor = image_name.split('_')[0][2:]
    toa_image = ee.Image(f'LANDSAT/LC{sensor}/C02/T1_TOA/{image_name}')

    # collection
    collect = ee.ImageCollection(f'USFS/GTAC/LCMS/v2023-9').filterDate(str(date), str(date+1)).first()
    # if collect.projection().crs().getInfo() != 'EPSG:4326':
    collect = collect.reproject(toa_image.select('B2').projection()).reproject('EPSG:4326', scale=SCALE)

    # get data
    n_nodatas = []
    for layer, key in zip(['Land_Cover', 'Land_Use'], ['landcover', 'landuse']):
        # get landcover and landuse masks
        lc = collect.select(layer)

        # crop image
        region = ee.Geometry.Polygon([[coords[0][0], coords[1][1]], 
                                    [coords[1][0], coords[1][1]],
                                    [coords[1][0], coords[0][1]],
                                    [coords[0][0], coords[0][1]],
                                    [coords[0][0], coords[1][1]]]) 
        
        feature = lc.toArray().sampleRectangle(region, defaultValue=0)
        image = np.array(feature.getInfo()['properties']['array'])
        org_shapes = image.shape[:2]
        if image.shape[0] != CROP_SIZE or image.shape[1] != CROP_SIZE:
            image = cv2.resize(image.astype(np.uint8), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
            image = image[:,:,np.newaxis]
        n_nodata = np.sum(image==0)
        n_nodatas.append(n_nodata)

        # save tif
        tif_path_lc = os.path.join(fpdir, f'usfs_{key}_{date}.tif')
        save_geotiff(image.astype(np.uint8), coords, tif_path_lc)
    return True, org_shapes, n_nodatas[0], n_nodatas[1]


def get_sbtn_lc_masks_from_ee(coords, fpdir, image_name, date=None):  
    # image
    sensor = image_name.split('_')[0][2:]
    toa_image = ee.Image(f'LANDSAT/LC{sensor}/C02/T1_TOA/{image_name}')

    # get landcover and landuse masks
    lc = ee.Image('WRI/SBTN/naturalLands/v1/2020').select('classification')
    # if lc.projection().crs().getInfo() != 'EPSG:4326':
    lc = lc.reproject(toa_image.select('B2').projection()).reproject('EPSG:4326', scale=SCALE)

    # crop image
    region = ee.Geometry.Polygon([[coords[0][0], coords[1][1]], 
                                  [coords[1][0], coords[1][1]],
                                  [coords[1][0], coords[0][1]],
                                  [coords[0][0], coords[0][1]],
                                  [coords[0][0], coords[1][1]]]) 
    
    feature = lc.toArray().sampleRectangle(region, defaultValue=0)
    image = np.array(feature.getInfo()['properties']['array'])
    org_shapes = image.shape[:2]
    if image.shape[0] != CROP_SIZE or image.shape[1] != CROP_SIZE:
        image = cv2.resize(image.astype(np.uint8), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        image = image[:,:,np.newaxis]
    n_nodata = np.sum(image==0)
    
    # save tif
    tif_path_lc = os.path.join(fpdir, 'sbtn_landcover_2020.tif')
    save_geotiff(image.astype(np.uint8), coords, tif_path_lc)
    return True, org_shapes, n_nodata


def worker(args):
    ee.Initialize()
    # lst, save_dir, dataset, date, name_strs = args
    df, save_dir, dataset, date = args
    fdir_data = os.path.join(save_dir, 'data')
    n_sam = len(df) # len(lst)
    err_meg, success_meg = [], []
    for i in range(n_sam):
        img_name = df.image_name.iloc[i]
        # ind = lst[i]
        # img_id = int(ind)
        img_id = df.id.iloc[i]
        ind = f"{img_id:07d}"
        fp = os.path.join(fdir_data, ind)
        # crop image data
        # coords, center_coords = crop_image_with_rasterio(fp, name_strs, CROP_SIZE)
        fp_sr = os.path.join(fp, 'sr.tif')
        with rasterio.open(fp_sr) as dst:
            bbox = dst.bounds
            center_coords = dst.transform * (dst.width//2, dst.height//2)
        coords = [[bbox.left, bbox.top], [bbox.right, bbox.bottom]]

        # get mask data
        try:
            if dataset == 'nlcd':
                outs = get_nlcd_lc_masks_from_ee(coords, fp, img_name, date=date)
            elif dataset == 'usfs':
                outs = get_usfs_lc_masks_from_ee(coords, fp, img_name, date=date)
            elif dataset == 'sbtn':
                outs = get_sbtn_lc_masks_from_ee(coords, fp, img_name)
        except Exception as e:
            outs = [False, e]
        
        # messages
        if outs[0]:
            success_meg.append([img_id, center_coords[0], center_coords[1], coords]+[out for out in outs[1:]])
        else:
            err_meg.append([img_id, center_coords[0], center_coords[1], outs[-1]])
    
    return {'err_meg': err_meg, 'success_meg': success_meg,}


if __name__ == '__main__':
    # dataset = 'usfs' # 'nlcd'
    # layer = 'Land_Use' # 'impervious'
    # ee.Authenticate(auth_mode='notebook') 
    n_process = 20
    save_dir_ = 'data/WeakFT'
    
    for dataset in ['usfs']:
        if dataset == 'nlcd':
            save_dir = os.path.join(save_dir_, 'nlcd')
            date = 2019
        elif dataset == 'usfs':
            save_dir = os.path.join(save_dir_, 'usfs')
            date = 2023
        elif dataset == 'sbtn':
            save_dir = os.path.join(save_dir_, 'sbtn')
            date = None
        else:
            raise ValueError('Invalid dataset!')

        # data directory and successful ids
        # data_dir = os.path.join(save_dir, 'data')
        # ids = os.listdir(data_dir)
        # print(f'{dataset}|Total {len(ids)} items to download!')
        fp_csv = os.path.join(save_dir, 'Downloaded_file_info.csv')
        df = pd.read_csv(fp_csv, header=0)

        # number of cores
        n_CPU = os.cpu_count()
        num_workers = min(n_CPU, n_process)
        err_meg, success_meg = [], []
        if num_workers>1:
            # Create a pool of worker processes
            print(f'{num_workers} cores out of {n_CPU} are used!')

            # Separate the dataframe into num_workers parts
            # sub_lists = separate_list(ids, num_workers)
            sub_dfs = separate_df(df, num_workers)
            # data = [(sub_lists[i], save_dir, dataset, date, name_strs) for i in range(num_workers)]
            data = [(sub_dfs[i], save_dir, dataset, date) for i in range(num_workers)]

            with multiprocessing.Pool(processes=num_workers) as p:
                results = p.map(worker, data)
            
            for r in results:
                err_meg += r['err_meg']
                success_meg += r['success_meg']
        else:
            print('No parallel processes are triggered!')
            # r = worker((ids, save_dir, dataset, date, name_strs))
            r = worker((df, save_dir, dataset, date))
            err_meg += r['err_meg']
            success_meg += r['success_meg']

        # Save the error and success messages
        fp_err = os.path.join(save_dir, f'Mask_error_center_coords_usa_{dataset}.csv')
        fp_success = os.path.join(save_dir, f'Mask_success_center_coords_usa_{dataset}.csv')
        if len(err_meg)>0:
            df_err = pd.DataFrame(err_meg, columns=['id', 'longitude', 'latitude', 'error'])
            df_err.to_csv(fp_err, index=False)
            print(f'{dataset}|Error messages are saved at {fp_err} ({len(err_meg)}/{len(df)} items)')
        if len(success_meg)>0:
            if dataset == 'sbtn':
                column_names = ['id', 'longitude', 'latitude', 'coords', 'org_shapes', 'n_nodata_lc']
            else:
                column_names = ['id', 'longitude', 'latitude', 'coords', 'org_shapes', 'n_nodata_lc', 'n_nodata_lu']
            df_success = pd.DataFrame(success_meg, columns=column_names)
            df_success.to_csv(fp_success, index=False)
            print(f'{dataset}|Success messages are saved at {fp_success} ({len(success_meg)}/{len(df)} items)')


