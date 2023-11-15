import sys
sys.path.append('/Users/earth/Desktop/vscode_anaconda/components')

from components.logger import init_logger
from components.helper_download import date_range
import logging
import yaml
import os
from glob import glob
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp
from multiprocessing import Manager 

import h5py
import netCDF4 as nc
import rioxarray
from rasterio.enums import Resampling
import time

from datetime import datetime, timedelta
from typing import Generator

# Variable / Parameter Configuration
THREADS:int = -1
if(THREADS == -1):
    THREADS = mp.cpu_count()

COLUMNS:dict = {
    'longitude': 'lon',
    'latitude': 'lat',
    'values': ['precipitationCal', 'precipitationUncal']
}

BOUNDING_BOX:dict = {
    'lat_range': [5.4, 20.6],
    'lon_range': [97.1, 106.0]
}

SOURCE_PATH:str = './IMERG'
AGGREGATED_PATH:str = './IMERG_Processed_Multiband'
GEOTIFF_PATH:str = './IMERG_Geotiff'
TIMEZONE:str = '+7'


def date_range(start_date:datetime, stop_date:datetime) -> Generator[datetime, None, None]:
    for n in range(int((stop_date - start_date + timedelta(1)).days)):
        yield start_date + timedelta(n)


def _get_filelist(timezone:str, target_day:datetime):
    source_list = []

    for i in range(0, 24):
        d1 = target_day + timedelta(hours=int(timezone), minutes=(i * 60))
        d2 = target_day + timedelta(hours=int(timezone), minutes=(i * 60)+30)
        # print(f"./data/imerg/{d1.year}/{d1.strftime('%j')}/*{d1.strftime('%Y%m%d-S%H%M')}*.HDF5")
        f1 = glob(f"{SOURCE_PATH}/{d1.year}/{d1.strftime('%j')}/*{d1.strftime('%Y%m%d-S%H%M')}*.HDF5")
        f2 = glob(f"{SOURCE_PATH}/{d2.year}/{d2.strftime('%j')}/*{d2.strftime('%Y%m%d-S%H%M')}*.HDF5")
        assert len(f1) == 1, f"{len(f1)} There is no file '{d1.strftime('%Y%m%d-S%H%M')}'. Therefore, process of date '{target_day}' with timezone {timezone} is not possible."
        assert len(f2) == 1, f"{len(f2)} There is no file '{d2.strftime('%Y%m%d-S%H%M')}'. Therefore, process of date '{target_day}' with timezone {timezone} is not possible."
        source_list.append(f1[0])
        source_list.append(f2[0])
    # logger.debug(f"{source_list=}")
    assert len(source_list) == 48, f"There should be 48 files but only {len(source_list)} is found. {source_list}"
    
    return source_list

def _aggregate(target_day:datetime, lock, destination:str, netcdf_filename:str) -> None:
    # Prepare parameter
    lon_range = BOUNDING_BOX['lon_range']
    lat_range = BOUNDING_BOX['lat_range']
    column_lat = COLUMNS['latitude']
    column_lon = COLUMNS['longitude']
    column_values = COLUMNS['values']

    # Prepare Destination
    if(os.path.exists(destination) == False):
        lock.acquire()
        if(os.path.exists(destination) == False):
            os.makedirs(destination)
        lock.release() 


    # IMERG_YYYYMMDD
    target_filename = os.path.join(destination, netcdf_filename)

    # Get list of files to aggregate
    raw_files = _get_filelist(timezone=TIMEZONE, target_day=target_day)

    h = h5py.File(raw_files[0], 'r')
    # 0. Bound to the target area
    bound_lon = np.logical_and(np.array(h['Grid'][column_lon]) >= lon_range[0], np.array(h['Grid'][column_lon]) <= lon_range[1])
    bound_lat = np.logical_and(np.array(h['Grid'][column_lat]) >= lat_range[0], np.array(h['Grid'][column_lat]) <= lat_range[1])

    ### Prepare NetCDF file
    ds = nc.Dataset(target_filename, 'w', format='NETCDF4')
    lat = ds.createDimension('lat', sum(bound_lat))
    lon = ds.createDimension('lon', sum(bound_lon))
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    values = dict()
    for column_value in column_values:
        values[column_value] = ds.createVariable(column_value, 'u2', ('lat', 'lon',))
    # values.units = 'Unknown'
    lats[:] = list(h['Grid']['lat'][bound_lat])
    lons[:] = list(h['Grid']['lon'][bound_lon])
    # 0. Bound to the target area
    data = dict()
    for column_value in column_values:
        bounded = h['Grid'][column_value][0][bound_lon]
        bounded = bounded[:, bound_lat]
        data[column_value] = np.zeros(bounded.shape)
    h.close()
    # data = bounded.copy()
    # data.astype(np.uint16)
    # np.ceil(int(data[idx_lat][idx_lon] * 10)/2)
    # data[bounded > 0] += np.ceil( int(bounded[bounded > 0] * 10) / 2 )

    # 2. Aggregate to 1 day
    for f in raw_files:
        # if(f == raw_files[0]): continue
        h = h5py.File(f,'r')
        for column_value in column_values:
            bounded = h['Grid'][column_value][0][bound_lon]
            bounded = bounded[:, bound_lat]
            temp = bounded.copy()
            # temp.astype(np.int32)
            # a = (temp[temp > 0] * 10).astype(np.int32)
            # b = a / 2
            # temp[bounded > 0] += np.ceil( b ).astype(np.int32)

            # temp += temp
            # temp[temp > 0] = np.ceil(temp[temp > 0] / 2)

            # if(sum(sum(temp < 0)) > 0): 
            #     logger.log(f"{target_filename} has data < 0")
            data[column_value] += temp
        h.close()

    # data = data * 10
    # data = data.astype(np.uint16)
    # data[data > 0] = np.ceil(  data[data>0] / 2  )
    # data = data.astype(np.uint16)
    for column_value in column_values:
        sel = data[column_value]
        sel = np.ceil((sel * 10).astype(np.int16) / 2)
        sel[sel < 0] = 29999
        sel = sel.astype(np.uint16)
        values[column_value][:, :] = np.expand_dims(sel.T, 0)
    ds.close()

def _convert_to_geotiff(target_day:datetime, full_path:str):
    # year,doy = _get_year_doy(path)
    year = target_day.year
    filename_with_ext = os.path.split(full_path)[-1]
    filename = os.path.splitext(filename_with_ext)[0]
    rds = rioxarray.open_rasterio(full_path, decode_times=False)
    rds.rio.write_crs('epsg:4326', inplace=True)

    # 1. Interpolate from 10KM (0.1◦) to 1KM (0.01◦)
    upscale_factor = 10
    new_width = rds.rio.width * upscale_factor
    new_height = rds.rio.height * upscale_factor
    xds_upsampled = rds.rio.reproject(
        rds.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.cubic,
    )

    # units, reference_date = rds.time.attrs['units'].split('since')
    # rds['Time'] = pd.date_range(start=reference_date, periods=rds.sizes['Time'], freq='MS')
    # rds.rio.set_nodata(29999, inplace=True)
    # folder_10km = f'./{GEOTIFF_PATH}/{year}/10KM'
    # folder_1km = f'./{GEOTIFF_PATH}/{year}/1KM'
    folder_10km = os.path.join(GEOTIFF_PATH, str(year), '10KM')
    folder_1km = os.path.join(GEOTIFF_PATH, str(year), '1KM')
    if(os.path.exists( folder_10km ) == False):
        os.makedirs(folder_10km)
    if(os.path.exists( folder_1km ) == False):
        os.makedirs(folder_1km)

    rds.squeeze().rio.to_raster(f'{folder_10km}/{filename}.tiff')
    xds_upsampled.squeeze().rio.to_raster(f'{folder_1km}/{filename}.tiff')
    rds.close()


def task(unprocess_date:datetime, lock) -> None:
    destination = os.path.join(AGGREGATED_PATH,str(unprocess_date.year))
    netcdf_filename = f'IMERG_{unprocess_date.strftime("%Y%m%d")}_{TIMEZONE}.nc'
    full_path = os.path.join(destination, netcdf_filename)
    try:
        # logger.info(f"Processing={unprocess_date}")
        # 2. Aggregate to 1 day
        _aggregate(unprocess_date, lock, destination, netcdf_filename)
        # logger.info(f"Aggregated file={netcdf_filename}")
        # 3. Save the file as GeoTiff
        _convert_to_geotiff(unprocess_date, full_path)
    except KeyboardInterrupt:
        print(f"Interrupt: {full_path}")
        os.remove(full_path)
    except Exception as e:
        # logger.error(f"{e=}\nError from '{unprocess_date}' with timezone={TIMEZONE}")
        os.remove(full_path)

def preprocess(start_date:datetime, stop_date:datetime) -> None:
    # get a list of date in the raw gsmap dataset 
    # date_range(start_date:datetime, stop_date:datetime)

    # temp:list[str] = glob(f"{SOURCE_PATH}/*/*")
    # list_of_rawdates:list[datetime] =  [datetime.strptime(f[-8:], '%Y/%j') for f in temp ]
    list_of_rawdates:list[datetime] =  list(date_range(start_date=start_date, stop_date=stop_date))
    # get a list of date that already preprocessd
    temp:list[str] = glob(f'{AGGREGATED_PATH}/*/*.nc')
    list_of_preprocessd:list[datetime] = [ datetime.strptime(os.path.split(f)[1][6:14],'%Y%m%d') for f in temp]

    # logger.debug(f"list_of_rawdates={list_of_rawdates}")
    # logger.debug(f"list_of_preprocessd={list_of_preprocessd}")
    # logger.info(f"Found list_of_rawdates={len(list_of_rawdates)} and list_of_preprocessd={len(list_of_preprocessd)}")

    list_of_unprocesses = list(set(list_of_rawdates).difference(list_of_preprocessd))
    # if(len(list_of_unprocesses) == 0):
        # Nothing to process
        # logger.info('Nothing to do')
    # else:
    list_of_unprocesses.sort()
    # logger.info(f"list_of_unprocesses={list_of_unprocesses}")
    manager = Manager()
    lock = manager.Lock()
    pool = mp.Pool(THREADS)
    try:
        results = [pool.apply_async( task, args=(unprocess, lock) ) for unprocess in list_of_unprocesses]
        results = [result.get() for result in results]
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    start_date = datetime.strptime('02-01-2023', '%d-%m-%Y')
    stop_date = datetime.strptime('02-01-2023', '%d-%m-%Y')

    preprocess(start_date, stop_date)
    
    # timezone= '+7'

    # bounding_box = {
    #     'lat_range': [5.4, 20.6],
    #     'lon_range': [97.1, 106.0]
    # }

    # columns = {
    #     'longitude': 'lon',
    #     'latitude': 'lat',
    #     'values': ['precipitationCal', 'precipitationUncal']
    # }

    # source_path= './IMERG'
    # aggregated_path= './IMERG_Processed'
    # geotiff_path= './IMERG_Geotiff'
    # threads= '-1'


    # THREADS:int = -1
    # if(THREADS == -1):
    #     THREADS = mp.cpu_count()

    # COLUMNS:dict = {
    #     'longitude': 'lon',
    #     'latitude': 'lat',
    #     'values': ['precipitationCal', 'precipitationUncal']
    # }

    # BOUNDING_BOX:dict = {
    #     'lat_range': [5.4, 20.6],
    #     'lon_range': [97.1, 106.0]
    # }

    # SOURCE_PATH:str = './IMERG'
    # AGGREGATED_PATH:str = './IMERG_Processed_Multiband'
    # GEOTIFF_PATH:str = './IMERG_Geotiff'
    # TIMEZONE:str = '+7'