import re
import ee
import os
import numpy as np
import rasterio
from rasterio.transform import Affine, from_bounds
from rasterio.windows import Window


def read_coords(bbox_str):
    # Regular expression to extract the values of left, bottom, right, and top
    pattern = r'left=(-?\d+\.\d+), bottom=(-?\d+\.\d+), right=(-?\d+\.\d+), top=(-?\d+\.\d+)'

    # Find all matches
    matches = re.search(pattern, bbox_str)

    # If matches are found, extract them into variables
    left = float(matches.group(1))
    bottom = float(matches.group(2))
    right = float(matches.group(3))
    top = float(matches.group(4))

    coords = [[left, top], [right, bottom]]
    return coords


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


def separate_list(lst, n_process):
    '''
    separate df into sub dfs according to the number of processes for parallel processing
    '''
    N = len(lst)
    n = np.ceil(N/n_process).astype(int)

    sub_lists = []
    for i in range(n_process):
        start = i*n
        end = (i+1)*n
        if i==n_process-1:
            end = N
        sub_lists.append(lst[start:end])
    return sub_lists


def get_properties(image):
    properties = {}
    for property in image.propertyNames().getInfo():
        properties[property] = image.get(property)
    return ee.Dictionary(properties).getInfo()


def get_image_array(features, bands):
    img_toa = []
    for band in bands:
        img_ = np.atleast_3d(features['properties'][band]) # [::-1,:,:]
        # img_ = center_crop(img_, out_size=(H,W))
        img_toa.append(img_)
        # raster[band] = img_.astype('uint16')
    img_toa = np.concatenate(img_toa, axis=-1)
    return img_toa


def get_coords(features):
    coords = np.array(features['geometry']['coordinates'][0])  
    coords = [
        [coords[:, 0].min(), coords[:, 1].max()],
        [coords[:, 0].max(), coords[:, 1].min()]
    ]
    return coords


def save_geotiff(img, coords, out_path):
    height, width, channels = img.shape
    transform = from_bounds(coords[0][0], coords[1][1], coords[1][0], coords[0][1], width, height)
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


def crop_image_with_rasterio(pdir, name_strs, crop_size, delete=False):
    for name_str in name_strs:
        fp = os.path.join(pdir, f'{name_str}_origin.tif')
        fp_save = os.path.join(pdir, f'{name_str}.tif')
        # read and crop image
        with rasterio.open(fp) as dataset:
            # Get the image dimensions
            img_width = dataset.width
            img_height = dataset.height
            
            # Calculate the center of the image
            center_x = img_width // 2
            center_y = img_height // 2
            
            # Calculate the crop window
            left = max(0, center_x - crop_size // 2)
            top = max(0, center_y - crop_size // 2)
            
            # Ensure the crop does not exceed image bounds
            right = min(img_width, left + crop_size)
            bottom = min(img_height, top + crop_size)
            
            # Create a Window from the calculated bounds
            window = Window(left, top, right - left, bottom - top)

            # resave
            kwargs = dataset.meta.copy()
            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, dataset.transform)})

            with rasterio.open(fp_save, 'w', **kwargs) as dst:
                dst.write(dataset.read(window=window))
                bbox = dst.bounds
                center_coords = dst.transform * (dst.width//2, dst.height//2)

        if delete: os.remove(fp)
        
    coords = [[bbox.left, bbox.top], [bbox.right, bbox.bottom]]

    return coords, center_coords


def center_crop(img, out_size):
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = int((image_height - crop_height + 1) * 0.5)
    crop_left = int((image_width - crop_width + 1) * 0.5)
    return img[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width]


def adjust_coords(coords, old_size, new_size):
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [coords[0][0] + ((xoff + new_size[1]) * xres), coords[0][1] - ((yoff + new_size[0]) * yres)]
    ]