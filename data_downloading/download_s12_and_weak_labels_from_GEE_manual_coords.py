import os
import ee
import cv2
import multiprocessing
from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils import save_geotiff, adjust_coords, center_crop, get_properties, separate_df


H, W = 264, 264
RADIUS = 1320
CLOUD = 10
BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
BANDS_2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
BUFF = 50


def get_overlap_points_iran(ssl4eo_dots_csv, iran_shp, save_path):
    # Define the bounding box for Iran
    iran_bbox = {
        'min_lon': 44.0,
        'max_lon': 63.0,
        'min_lat': 25.0,
        'max_lat': 40.0
    }

    # Load the CSV file
    # ssl4eo_dots_csv = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/center_coords_ssl4eo_save_order.csv'
    data = pd.read_csv(ssl4eo_dots_csv)

    # Check if the coordinates are within the Iran bounding box
    def is_in_iran(lon, lat):
        return (iran_bbox['min_lon'] <= lon <= iran_bbox['max_lon']) and (iran_bbox['min_lat'] <= lat <= iran_bbox['max_lat'])

    # Apply the function to each row
    data['in_iran'] = data.apply(lambda row: is_in_iran(row.iloc[1], row.iloc[2]), axis=1)

    # Count the number of patches in Iran
    num_patches_in_iran = data['in_iran'].sum()
    print(f"Number of patches filtered by bounding box: {num_patches_in_iran}")

    # Load the shapefile for Iran's boundary
    # iran_shapefile = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/Iran_shp/irn_admbnda_adm0_unhcr_20190514.shp'
    iran_boundary = gpd.read_file(iran_shp)

    # Convert the DataFrame to a GeoDataFrame
    # Rename the columns to 'longitude' and 'latitude' for consistency
    data.rename(columns={data.columns[1]: 'longitude', data.columns[2]: 'latitude'}, inplace=True)

    # Convert the DataFrame to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    geo_data = gpd.GeoDataFrame(data, geometry=geometry)

    # Check if the points are within the Iran boundary
    geo_data['in_iran'] = geo_data.geometry.apply(lambda x: iran_boundary.contains(x).any())

    # Count the number of patches in Iran
    num_patches_in_iran = geo_data['in_iran'].sum()

    print(f'Number of patches in Iran: {num_patches_in_iran}')

    # Save the filtered data to a new CSV file
    # filtered_csv_file = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/center_coords_ssl4eo_iran.csv'
    geo_data[geo_data['in_iran']].to_csv(save_path, index=False)

    print(f'Filtered data saved to {save_path}')

    return geo_data


def clip_img_with_centroid_s2(lon, lat, START_DATE, END_DATE=None):
    '''
    clip s2 images on ee using bounding box and crs
    specifically, to further avoid co-registration problem, we double check the crs of fetched images
    only fetch from the images with the same crs as required
    in this case, the fetched images should be of exactly the same size as the target/DW-label images
    '''
    # s2 collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # Filter the collection to specific dates and region
    # filter_point = ee.Geometry.Point([lon, lat])
    # buffered_region = ee.Geometry.Point([lon, lat]).buffer(RADIUS+100).bounds() # sample region bound
    region = ee.Geometry.Point([lon, lat]).buffer(RADIUS).bounds() # sample region bound
    region_with_buffer = region.buffer(BUFF)
    if len(START_DATE)==4:
        if lat>=0:
            # northern hemisphere
            END_DATE = f"{START_DATE}-09-01"
            START_DATE = f"{START_DATE}-06-01"
        else:
            # southern hemisphere
            END_DATE = f"{START_DATE}-04-01"
            START_DATE = f"{START_DATE}-01-01"
    # convert to date string
    t_start = datetime.strptime(START_DATE, "%Y-%m-%d")
    t_end  = datetime.strptime(END_DATE, "%Y-%m-%d")
    s2 = s2.filterDate(t_start, t_end).filterBounds(region_with_buffer)

    # Fileter cloud coverage
    s2 = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD))

    # Grab the image one by one in the filtered collection
    s2 = s2.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    # n_s2 = s2.size().getInfo()
    # print(f"Number of S2 images: {n_s2}")
    
    # select 
    s2_image = s2.select(*BANDS) #.sampleRectangle(region)

    # # Resample the data so that the bands all map 1 pixel -> 10m. We'll use B2 (red)
    # # for a reference projection.
    s2_image = s2_image.toFloat().resample('bilinear').reproject(
        s2_image.select('B2').projection()).reproject('EPSG:4326', scale=10)
    s2_image = s2_image.sampleRectangle(region)
    metadata = get_properties(s2)
    features = s2_image.getInfo()
    
    # clip image
    img = []
    for band in BANDS:
        img_ = np.atleast_3d(features['properties'][band]) 
        img.append(img_)
    img = np.concatenate(img, axis=-1)
    img = center_crop(img, out_size=(H,W))
    coords = np.array(features['geometry']['coordinates'][0])
    coords = [
        [coords[:, 0].min(), coords[:, 1].max()],
        [coords[:, 0].max(), coords[:, 1].min()]
    ]

    # get new coords
    band = BANDS[0]
    old_size = (len(features['properties'][band]), len(features['properties'][band][0]))
    new_size = (H,W)
    coords = adjust_coords(coords, old_size, new_size)    

    return img, metadata, coords
    

def get_single_s2_data_from_ee(lon, lat, START_DATE, END_DATE):  
    # fetch data
    try:
        # lowest cloud coverage in the summer season
        image, meta, coords = clip_img_with_centroid_s2(lon, lat, START_DATE, END_DATE)
        return image, meta, coords
    except Exception as e:
        error_msg = f"S2 Pulling error: {e}"
        return error_msg
    

def clip_s1_accord_to_s2(coords, s2_meta):
    '''
    clip s2 images on ee using bounding box and crs
    specifically, to further avoid co-registration problem, we double check the crs of fetched images
    only fetch from the images with the same crs as required
    in this case, the fetched images should be of exactly the same size as the target/DW-label images
    '''
    # s1 collection
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD')

    # Get the date of the s2_example image
    date = ee.Date(s2_meta['system:time_start'])

    # Define the date range for filtering
    start_date = date.advance(-1, 'day')
    end_date = date.advance(15, 'day')
    # region
    region = ee.Geometry.Polygon([[coords[0][0], coords[1][1]], 
                                  [coords[1][0], coords[1][1]],
                                  [coords[1][0], coords[0][1]],
                                  [coords[0][0], coords[0][1]],
                                  [coords[0][0], coords[1][1]]]) 
    region_with_buffer = region.buffer(BUFF)

    # Filter the s1 collection by date range
    s1 = s1.filterDate(start_date, end_date).filterBounds(region_with_buffer)

    # Filter the content
    s1 = s1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    s1 = s1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    s1 = s1.sort('system:time_start')
    
    # Get the length of the s1 collection
    s1_images = s1.toList(s1.size())
    s1_length = s1_images.length().getInfo()
    # print(f"Number of S1 images: {s1_length}")
    
    # Get the image by order
    if s1_length == 0:
        raise ("No S1 images found for the given date range.")
    else:
        for i in range(s1_length):
            try:
                s1_image = ee.Image(s1_images.get(i))
                feature = s1_image.select('VV', 'VH')
                # feature = feature.toFloat().resample('bilinear').reproject('EPSG:4326', scale=10)
                feature = feature.reproject('EPSG:4326', scale=10)
                feature = feature.toArray().sampleRectangle(region, defaultValue=0)
                image = np.array(feature.getInfo()['properties']['array'])
                assert image.shape[:2]==(H,W), f"S1 image shape is {image.shape[:2]}"
                break
            except Exception as e:
                if i<s1_length-1:
                    continue
                else:
                    raise ValueError(f"Error in fetching S1 image with last error message: {e}")

    # image id
    img_id = s1_image.get('system:index').getInfo()
    
    return image, img_id


def get_single_s1_data_from_ee(coords, s2_meta):  
    # fetch data
    try:
        # current day + 15 days
        image, s1_imageid = clip_s1_accord_to_s2(coords, s2_meta)
        return image, s1_imageid
    except Exception as e:
        error_msg = f"Pulling error: {e}"
        return error_msg


def get_iran_lc_masks_from_ee(success_s2, fpdir):  
    # s1 collection
    lc = ee.Image('KNTU/LiDARLab/IranLandCover/V1').select('classification').reproject('EPSG:4326', scale=10)

    err_lc, success_lc = [], []
    for i in range(len(success_s2)):
        try:
            did = success_s2[i][0]
            coords = success_s2[i][2]
            # region
            region = ee.Geometry.Polygon([[coords[0][0], coords[1][1]], 
                                        [coords[1][0], coords[1][1]],
                                        [coords[1][0], coords[0][1]],
                                        [coords[0][0], coords[0][1]],
                                        [coords[0][0], coords[1][1]]]) 
            feature = lc.toArray().sampleRectangle(region, defaultValue=0)
            image = np.array(feature.getInfo()['properties']['array'])
            if image.shape[0]>=H and image.shape[1]>=W:
                success_lc.append([did, success_s2[i][1], coords, image.shape])
                image = image[:H, :W]
            else:
                raise ValueError(f"Landcover image shape is {image.shape[:2]}")
            tif_path_lc = os.path.join(fpdir, f'{did:07}', f'lc_{did:07}.tif')
            save_geotiff(image.astype(np.uint8), coords, tif_path_lc)
        except Exception as e:
            err_lc.append([did, f"Pulling error: {e}"])
    return err_lc, success_lc


def get_worldcover_lc_masks_from_ee(success_s2, fpdir, v):  
    # s1 collection
    lc = ee.ImageCollection(f'ESA/WorldCover/{v}').first().select('Map')
    if lc.projection().crs().getInfo() != 'EPSG:4326':
        crs_ = lc.projection().crs().getInfo()
        lc = lc.reproject('EPSG:4326', scale=10)
        print(f"Convert the crs of WorldCover from {crs_} to EPSG:4326.")
    
    err_lc, success_lc = [], []
    for i in range(len(success_s2)):
        try:
            did = success_s2[i][0]
            coords = success_s2[i][2]
            # region
            region = ee.Geometry.Polygon([[coords[0][0], coords[1][1]], 
                                        [coords[1][0], coords[1][1]],
                                        [coords[1][0], coords[0][1]],
                                        [coords[0][0], coords[0][1]],
                                        [coords[0][0], coords[1][1]]]) 
            feature = lc.toArray().sampleRectangle(region, defaultValue=0)
            image = np.array(feature.getInfo()['properties']['array'])
            if image.shape[0]>=H and image.shape[1]>=W:
                success_lc.append([did, success_s2[i][1], coords, image.shape])
                # image = image[:H, :W]
                if image.shape[0] != H or image.shape[1] != W:
                    image = cv2.resize(image.astype(np.uint8), (H,W), interpolation=cv2.INTER_NEAREST)
                    image = image[:,:,np.newaxis]
            else:
                raise ValueError(f"Landcover image shape is {image.shape[:2]}")
            tif_path_lc = os.path.join(fpdir, f'{did:07}', f'lc_{did:07}.tif')
            save_geotiff(image.astype(np.uint8), coords, tif_path_lc)
        except Exception as e:
            err_lc.append([did, f"Pulling error: {e}"])
    return err_lc, success_lc


def get_s12_for_each_split_tif(args):
    ee.Initialize()
    t0 = datetime.now().replace(microsecond=0)
    df, fpdir, process_id, dataset, START_DATE, END_DATE = args
    print(f'Start the {process_id}-th process ({df.shape[0]} samples)!')
    
    # get data properties
    n_sam = df.shape[0]
    
    # # # # # # # # # # get sentinel data from ee # # # # # # # # # #
    err_s2, err_s1 = [], []
    success_s2, success_s1 = [], []
    for i in range(n_sam):
        s2_success = False
        lon, lat = float(df.iloc[i,1]), float(df.iloc[i,2])
        did = df.iloc[i,0]
        # 1> get s2 from ee
        out_s2 = get_single_s2_data_from_ee(lon, lat, START_DATE, END_DATE)
        try:
            s2_image, s2_meta, s2_coords = out_s2
            uni = np.unique(s2_image)
            if len(uni)<13*9:
                err_msg = f'S2-{did}: Too many invalid values'
            else:
                # interpolat to target size
                err_msg = 0
                s2_imageid = s2_meta['system:index']
                s2_success = True
        except:       
            if type(out_s2) is str:
                err_msg = out_s2
            else:
                print(f"S2-{did}:unknown error with {type(out_s2)}-{out_s2}")
                err_msg = 'S2: unknown error!'
        # error message logging
        if s2_success:
            success_s2.append([did, s2_imageid, s2_coords])
        else:
            err_s2.append([did, err_msg])

        # 2> get s1 from ee
        s1_success = False
        if s2_success:
            out_s1 = get_single_s1_data_from_ee(s2_coords, s2_meta)
            if type(out_s1) is str:
                err_msg = out_s1
            else:
                s1_image, s1_imageid = out_s1
                uni = np.unique(s1_image)
                if len(uni)<2*9:
                    err_msg = 'S1: Too many invalid values'
                else:
                    err_msg = 0
                    s1_success = True
        else:
            err_msg = 'S1: due to S2 data not fetched!'
        # error message logging
        if s1_success:
            success_s1.append([did, s2_imageid, s2_coords])
        else:
            err_s1.append([did, err_msg])
        
        # # # # # saving images # # # # #
        # save s2 images
        if s2_success:
            fdir = os.path.join(fpdir, f'{df.iloc[i,0]:07}', 's2')
            Path(fdir).mkdir(parents=True, exist_ok=True)  
            tif_path = os.path.join(fdir, f'{s2_imageid}.tif')
            save_geotiff(s2_image.astype(np.uint16), s2_coords, tif_path)
            # save s1 images
        if s1_success:
            fdir = os.path.join(fpdir, f'{df.iloc[i,0]:07}', 's1')
            Path(fdir).mkdir(parents=True, exist_ok=True)
            tif_path = os.path.join(fdir, f'{s1_imageid}.tif')
            save_geotiff(s1_image.astype(np.float32), s2_coords, tif_path)
        
        # progress
        if i>0 and i%50==0:
            t1 = datetime.now().replace(microsecond=0)
            print(f'PID-{process_id}: {i}/{n_sam} samples have been processed using {t1-t0}s!')
    
    # # # # # # # # # get landcover masks from ee # # # # # # # # # #
    t1 = datetime.now().replace(microsecond=0)
    if dataset == 'Iran':
        print(f'PID-{process_id}: Start fetching landcover masks for {dataset} (fetching s1 and s2 takes {t1-t0}s)!')
        err_lc, success_lc = get_iran_lc_masks_from_ee(success_s2, fpdir)
    elif 'worldcover' in dataset:
        print(f'PID-{process_id}: Start fetching landcover masks for {dataset} (fetching s1 and s2 takes {t1-t0}s)!')
        v = dataset.split('_')[-1]
        err_lc, success_lc = get_worldcover_lc_masks_from_ee(success_s2, fpdir, v)
     
    # print fetching results
    t1 = datetime.now().replace(microsecond=0)
    print(f'Finish the {process_id}-th process using {t1-t0}s (with {len(err_s1)} S1 and {len(err_s2)} S2 failing out of {df.shape[0]} samples)!')
    
    return {'err_s1': err_s1, 'err_s2': err_s2, 'err_lc': err_lc, 
            'success_s1': success_s1, 'success_s2': success_s2, 'success_lc': success_lc}



if __name__ == '__main__':
    dataset = 'Iran'    
    n_process = 1
    save_dir = f'/p/scratch/hai_ws4eo/liu_ch/data/WeakFT/{dataset}'

    # # # # # # settings # # # # # #
    if dataset == 'Iran':
        print('Start fetching data for Iran!')
        get_iran_spots = False
        csv_path = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/centroids_csv/center_coords_ssl4eo_iran.csv'
        
        # date
        START_DATE = '2017'#-06-01'
        END_DATE   = '2017'#-09-01'

        # read center points    
        if get_iran_spots:
            ssl4eo_dots_csv = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/center_coords_ssl4eo_save_order.csv'
            iran_shp = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/Iran_shp/irn_admbnda_adm0_unhcr_20190514.shp'
            geo_data = get_overlap_points_iran(ssl4eo_dots_csv, iran_shp, csv_path)
        else:
            geo_data = pd.read_csv(csv_path, header=None)
        
    elif dataset in ['worldcover_v100', 'worldcover_v200']:
        print('Start fetching data for WorldCover!')
        # centroids csv
        csv_path = '/p/project1/hai_ws4eo/liu_ch/Weak_FT/segment-anything-2/data_downloading/centroids_csv/center_coords_subset_worldcover_from_ssl4eo.csv'
        geo_data = pd.read_csv(csv_path, header=None)

        # date
        START_DATE = '2020' if 'v100' in dataset else '2021'
        END_DATE   = START_DATE



    # # # # # # downloading # # # # # #
    # check whether the storage directory exists
    save_dir_data = os.path.join(save_dir, 'data')
    Path(save_dir_data).mkdir(parents=True, exist_ok=True)
        
    # 4 - get core number
    # get number of cpu cores
    n_CPU = multiprocessing.cpu_count()  
    num_processes = min(n_CPU, n_process)
    
    # 5 - fetch data from ee
    get_s12_for_each_split = get_s12_for_each_split_tif
    err_s1, err_s2, err_lc, success_s1, success_s2, success_lc = [], [], [], [], [], []
    if num_processes>1:
        # Create a pool of worker processes
        print(f'{num_processes} cores out of {n_CPU} are used!')

        # separate dfs for parallel processing
        sub_dfs = separate_df(geo_data, num_processes)
        
        # prepare data
        data = [(sub_dfs[k], save_dir_data, k, dataset, START_DATE, END_DATE) for k in range(len(sub_dfs))]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(get_s12_for_each_split, data)
        # record failed files
        for r in results: 
            err_s1 += r['err_s1']
            err_s2 += r['err_s2']
            err_lc += r['err_lc']
            success_s1 += r['success_s1']
            success_s2 += r['success_s2']
            success_lc += r['success_lc']
    else:
        # single process
        print('No parallel processes are triggered!')
        r = get_s12_for_each_split((geo_data, save_dir_data, 0, dataset, START_DATE, END_DATE))
        err_s1 += r['err_s1']
        err_s2 += r['err_s2']
        err_lc += r['err_lc']
        success_s1 += r['success_s1']
        success_s2 += r['success_s2']
        success_lc += r['success_lc']
    
    # 6 - output failed file names
    for i, err in zip(['s1', 's2', 'lc'], [err_s1, err_s2, err_lc]):
        if len(err)>0:
            df_err = pd.DataFrame(err, columns=['img_id', 'error_message'])
            df_err.to_csv(os.path.join(save_dir, f'Error_message_{i}.csv'), index=False)
            print(f'{i}: Total {df_err.shape[0]} data failed to fetch!')
        else:
            print(f'{i}: All data have been successfully downloaded!')
    for i, success in zip(['s1', 's2', 'lc'], [success_s1, success_s2, success_lc]):
        if len(success)>0:
            if i == 'lc':
                df_suc = pd.DataFrame(success, columns=['img_id', 'image_name', 'coords', 'init_image_shape'])
            else:
                df_suc = pd.DataFrame(success, columns=['img_id', 'image_name', 'coords'])
            df_suc.to_csv(os.path.join(save_dir, f'Downloaded_file_info_{i}.csv'), index=False)
            print(f'{i}: Total {len(success)} data have been successfully downloaded!')
        else:
            print(f'{i}: No data has been successfully downloaded!')
    
    
        
    
