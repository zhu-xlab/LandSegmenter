import multiprocessing
import os
from pathlib import Path
import numpy as np
from datetime import datetime

import ee
import pandas as pd
from datetime import datetime

from utils import separate_df, get_properties, get_image_array, get_coords, save_geotiff, crop_image_with_rasterio
from download_lc_for_landsat_from_GEE_rasterio_coords import get_nlcd_lc_masks_from_ee, get_usfs_lc_masks_from_ee, get_sbtn_lc_masks_from_ee

CROP_SIZE = 264 
H, W = 264, 264
RADIUS = 3960
CLOUD = 10
BANDS_TOA = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']
BANDS_SR  = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
BUFF = 90
SCALE = 30


def clip_img_with_centroid_landsat(lon, lat, START_DATE, END_DATE=None, sensor=8):
    '''
    clip s2 images on ee using bounding box and crs
    specifically, to further avoid co-registration problem, we double check the crs of fetched images
    only fetch from the images with the same crs as required
    in this case, the fetched images should be of exactly the same size as the target/DW-label images
    '''
    # s2 collection
    toa = ee.ImageCollection(f'LANDSAT/LC{sensor:02d}/C02/T1_TOA')
    # sr  = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')

    # Filter the collection to specific dates and region
    region = ee.Geometry.Point([lon, lat]).buffer(RADIUS).bounds()
    if len(START_DATE)==4:
        if lat>=0:
            # northern hemisphere
            END_DATE = f"{START_DATE}-11-01"
            START_DATE = f"{START_DATE}-05-01"
        else:
            # southern hemisphere
            END_DATE = f"{START_DATE}-04-01"
            START_DATE = f"{START_DATE}-01-01"
    # convert to date string
    t_start = datetime.strptime(START_DATE, "%Y-%m-%d")
    t_end  = datetime.strptime(END_DATE, "%Y-%m-%d")
    toa = toa.filterDate(t_start, t_end).filterBounds(region)
    # print(toa.size().getInfo())

    # Fileter cloud coverage
    toa = toa.filter(ee.Filter.lt('CLOUD_COVER', CLOUD))
    # print(toa.size().getInfo())

    # Grab the image one by one in the filtered collection
    toa = toa.sort('CLOUD_COVER').first()
    img_id = toa.get('system:index').getInfo()
    
    # select 
    toa_image = toa.select(*BANDS_TOA) #.sampleRectangle(region)
    org_crs = toa_image.select(BANDS_TOA[1]).projection().crs().getInfo()

    # # Resample the data so that the bands all map 1 pixel -> 10m. We'll use B2 (red)
    # # for a reference projection.
    toa_image = toa_image.resample('bilinear').reproject(
        toa_image.select(BANDS_TOA[1]).projection()).reproject('EPSG:4326', scale=SCALE)#.clip(region)
    # print(s2_image.projection().getInfo())
    metadata = get_properties(toa)
    metadata['system:crs'] = org_crs
    # cropping
    features = toa_image.sampleRectangle(region).getInfo()
    # get image array
    img_toa = get_image_array(features, BANDS_TOA)
    assert img_toa.shape[0]>=H and img_toa.shape[1]>=W, f"TOA image is too small: {img_toa.shape}"
    # coordinates
    coords = get_coords(features)

    # get sr data
    try: # try to get s2_sr
        sr_image = ee.Image(f'LANDSAT/LC{sensor:02d}/C02/T1_L2/{img_id}')
        sr = sr_image.select(BANDS_SR)
        sr = sr.toFloat().reproject('EPSG:4326', scale=SCALE)
        sr = sr.sampleRectangle(region).getInfo()
        img_sr = get_image_array(sr, BANDS_SR)
        assert img_toa.shape[:2]==img_sr.shape[:2], f"TOA and SR images have different sizes!"
        return img_toa, img_sr, metadata, coords
    except Exception as e:
        return e
    

def get_single_landsat_data_from_ee(lon, lat, START_DATE, END_DATE, sensor):  
    # fetch data
    try:
        # lowest cloud coverage in the summer season
        image_toa, image_sr, meta, coords = clip_img_with_centroid_landsat(lon, lat, START_DATE, END_DATE, sensor)
        return image_toa, image_sr, meta, coords
    except Exception as e:
        error_msg = f"Landsat Pulling error: {e}"
        return error_msg


def get_data_for_each_split(args):
    ee.Initialize()
    t0 = datetime.now().replace(microsecond=0)
    df, fpdir, process_id, START_DATE, END_DATE, sensor = args
    
    # get data properties
    n_sam = df.shape[0]
    
    # # # # # # # # # # get sentinel data from ee # # # # # # # # # #
    success_info, err_info = [], []
    success_mask, err_mask = [], []
    for i in range(n_sam):
        lon, lat = df.longitude.iloc[i], df.latitude.iloc[i]
        did = df.id.iloc[i]
        # 1> get image from ee
        out_img = get_single_landsat_data_from_ee(lon, lat, START_DATE, END_DATE, sensor)
        try:
            image_toa, image_sr, meta, coords = out_img
            uni_toa = np.unique(image_toa)
            uni_sr = np.unique(image_sr)
            if len(uni_toa)<13*9:
                err_msg = f'TOA-{did}: Too many invalid values'
            elif len(uni_sr)<7*9:
                err_msg = f'SR-{did}: Too many invalid values'
            else:
                # interpolat to target size
                err_msg = 0
                imageid = meta['system:index']
        except:       
            if type(out_img) is str:
                err_msg = out_img
            else:
                print(f"TOA-{did}:unknown error with {type(out_img)}-{out_img}")
                err_msg = 'TOA: unknown error!'

        # error message logging
        if err_msg == 0:
            for k, img, dt in zip(['toa', 'sr'], [image_toa, image_sr], [np.float32, np.uint16]):
                fdir = os.path.join(fpdir, f'{did:07}')
                Path(fdir).mkdir(parents=True, exist_ok=True)  
                tif_path = os.path.join(fdir, f'{k}_origin.tif')
                save_geotiff(img.astype(dt), coords, tif_path)
            success_info.append([did, imageid, coords, meta['system:crs'], image_toa.shape])
        else:
            err_info.append([did, lon, lat, err_msg])
        
        # 2> crop image with rasterio and update coords
        if err_msg == 0:
            coords, center_coords = crop_image_with_rasterio(fdir, ['toa', 'sr'], CROP_SIZE, delete=True)
        
        # 3> get mask data
        if err_msg == 0:
            try:
                if dataset == 'nlcd':         
                    outs = get_nlcd_lc_masks_from_ee(coords, fdir, imageid, date=START_DATE)
                elif dataset == 'usfs':
                    outs = get_usfs_lc_masks_from_ee(coords, fdir, imageid, date=START_DATE)
                elif dataset == 'sbtn':
                    outs = get_sbtn_lc_masks_from_ee(coords, fdir, imageid)
            except Exception as e:
                outs = [False, e]

            # messages
            if outs[0]:
                success_mask.append([did, center_coords[0], center_coords[1], coords]+[out for out in outs[1:]])
            else:
                err_mask.append([did, center_coords[0], center_coords[1], outs[-1]])

     
    # print fetching results
    t1 = datetime.now().replace(microsecond=0)
    print(f'Finish the {process_id}-th process using {t1-t0}s (with {len(success_mask)}/{len(success_info)}/{df.shape[0]} successful)!')
    
    return {'success_info': success_info, 'err_info': err_info, 'success_mask': success_mask, 'err_mask': err_mask}



if __name__ == '__main__':
    # download_mask = True
    for dataset in ['usfs', 'sbtn', 'nlcd']:  
        t0 = datetime.now().replace(microsecond=0)
        n_process = 70
        save_dir = f'/p/scratch/hai_dm4eo/liu_ch/data/WeakFT/{dataset}'

        # # # # # # settings # # # # # #
        if dataset == 'nlcd':
            # centroids csv
            csv_path = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/centroids_csv/center_coords_usa.csv'
            
            # date
            START_DATE = '2019'   # candidates: 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019
            END_DATE   = START_DATE

            # sensor
            sensor = 8
            
        elif dataset == 'usfs':
            # centroids csv
            csv_path = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/centroids_csv/center_coords_usa.csv'
            # '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/centroids_csv/center_coords_subset_usa_from_ssl4eo.csv'

            # date
            START_DATE = '2023'   # 1985-2023
            END_DATE   = START_DATE

            # sensor
            sensor = 9
        
        elif dataset == 'sbtn':
            # centroids csv
            csv_path = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/centroids_csv/center_coords_subset_sbtn_from_ssl4eo_l.csv'

            # date
            START_DATE = '2020'   # 1985-2023
            END_DATE   = START_DATE

            # sensor
            sensor = 8

        # # # # # # downloading # # # # # #
        geo_data = pd.read_csv(csv_path, header=0)
        print(f'Start fetching data for {dataset} (total:{len(geo_data)})!')
        
        # check whether the storage directory exists
        save_dir_data = os.path.join(save_dir, 'data')
        Path(save_dir_data).mkdir(parents=True, exist_ok=True)
            
        # 4 - get core number
        # get number of cpu cores
        n_CPU = multiprocessing.cpu_count()  
        num_processes = min(n_CPU, n_process)
        
        # 5 - fetch data from ee
        err_info, success_info = [], []
        err_mask, success_mask = [], []
        if num_processes>1:
            # Create a pool of worker processes
            print(f'{num_processes} cores out of {n_CPU} are used!')

            # separate dfs for parallel processing
            sub_dfs = separate_df(geo_data, num_processes)
            
            # prepare data
            data = [(sub_dfs[k], save_dir_data, k, START_DATE, END_DATE, sensor) for k in range(len(sub_dfs))]
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(get_data_for_each_split, data)
            # record failed files
            for r in results: 
                err_info += r['err_info']
                success_info += r['success_info']
                err_mask += r['err_mask']
                success_mask += r['success_mask']
        else:
            # single process
            print('No parallel processes are triggered!')
            r = get_data_for_each_split((geo_data, save_dir_data, 0, START_DATE, END_DATE, sensor))
            err_info = r['err_info']
            success_info = r['success_info']
            err_mask = r['err_mask']
            success_mask = r['success_mask']

        # 6 - output failed file names   
        if len(success_info)>0:
            df_suc = pd.DataFrame(success_info, columns=['id', 'image_name', 'coords', 'img_origin_crs', 'init_img_shape'])           
            df_suc.to_csv(os.path.join(save_dir, 'Downloaded_file_info.csv'), index=False)
            print(f'{dataset}|Total {len(success_info)}/{geo_data.shape[0]} image data have been successfully downloaded!')
        else:
            print(f'{dataset}|No image data has been successfully downloaded!')

        if len(err_info)>0:
            df_err = pd.DataFrame(err_info, columns=['id', 'longitude', 'latitude', 'error_message'])
            df_err.to_csv(os.path.join(save_dir, 'Error_message.csv'), index=False)
            print(f'{dataset}|Total {df_err.shape[0]}/{geo_data.shape[0]} image data failed to fetch!')
        else:
            print(f'{dataset}|All image data have been successfully downloaded!')

        # mask outputs        
        fp_err = os.path.join(save_dir, f'Mask_error_center_coords_usa_{dataset}.csv')
        fp_success = os.path.join(save_dir, f'Mask_success_center_coords_usa_{dataset}.csv')
        if len(err_mask)>0:
            df_err = pd.DataFrame(err_mask, columns=['id', 'longitude', 'latitude', 'error'])
            df_err.to_csv(fp_err, index=False)
            print(f'{dataset}|Error messages are saved at {fp_err} ({len(err_mask)}/{len(success_info)} items)')
        if len(success_mask)>0:
            if dataset == 'sbtn':
                column_names = ['id', 'longitude', 'latitude', 'coords', 'org_shapes', 'n_nodata_lc']
            else:
                column_names = ['id', 'longitude', 'latitude', 'coords', 'org_shapes', 'n_nodata_lc', 'n_nodata_lu']
            df_success = pd.DataFrame(success_mask, columns=column_names)
            df_success.to_csv(fp_success, index=False)
            print(f'{dataset}|Success messages are saved at {fp_success} ({len(success_mask)}/{len(success_info)} items)')
        
        # time used
        t1 = datetime.now().replace(microsecond=0)
        print(f'{dataset}|Total time used: {t1-t0}!')


    
    
        
    
