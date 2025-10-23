import os
import rasterio
import cv2
import numpy as np
from rasterio.transform import from_bounds
from tqdm import tqdm

def save_geotiff(img, out_path, left, bottom, right, top):
    height, width, channels = img.shape
    transform = from_bounds(left, bottom, right, top, width, height)
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


def resize_tif(fp, target_size=(264, 264)):
    # Open the original TIFF image using rasterio
    with rasterio.open(fp) as src:
        # Read the image data and metadata
        data = src.read().squeeze()  # data shape: (bands, height, width)
        # print(data.shape)
        
        if data.shape != target_size:
            # metadata = src.meta.copy()

            # Use cv2 to resize the image to the target size
            data = cv2.resize(data, target_size, interpolation=cv2.INTER_NEAREST)
            data = data[:, :, np.newaxis]  # Add the band dimension back

            # # Preserve the metadata (same transform, coordinate info, etc.)
            # metadata.update({
            #     "height": target_size[0],
            #     "width": target_size[1]
            # })
            fp_s2 = os.path.join(os.path.dirname(fp), 's2_l1c.tif')
            with rasterio.open(fp_s2) as src_s2:
                bbox = src_s2.bounds
            # bbox = src.bounds

            resave = True
        else:
            resave = False

    # Write the resized image back to the output path
    if resave:
        # fp = os.path.join(os.path.dirname(fp), 'lc_resized.tif')
        save_geotiff(data, fp, *bbox)

def resize_images_in_directory(input_directory, target_size=(264, 264)):
    for subdir in tqdm(os.listdir(input_directory), total=len(os.listdir(input_directory))):
        fp = os.path.join(input_directory, subdir, 'lc.tif')
        if os.path.isfile(fp):
            resize_tif(fp, target_size)
            # print(f"Resized and saved: {filename}")


if __name__ == '__main__':
    # Directory containing your TIFF images
    input_directory = "/p/scratch/hai_dm4eo/liu_ch/data/WeakFT/worldcover_v200/data"
    print(f"Resizing images in {input_directory}")
    resize_images_in_directory(input_directory)