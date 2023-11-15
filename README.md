* Prerequisite
- \> pip install netCDF4
- \> pip install h5py
- \> pip install rioxarray
- \> pip install rasterio

<br/>

- **Download IMERG HDF5 to ./IMERG**
    - EX: Download 01/02/2023
    - Path: ./IMERG/2023/001/XXXX.hdf5 (หลายๆตัว)

<br/>

- **To get a single NetCDF file (.nc) for that a particular day or days**
    - update: 
        - \> start_date = datetime.strptime('02-01-2023', '%d-%m-%Y') 
        - \> stop_date = datetime.strptime('02-01-2023', '%d-%m-%Y')
        - \> execute: python3 Preprocess.py
    - The output file will be located at ./IMERG_Processed_Multiband

<br/>

- **EXTRA: convert .nc file to .csv**
    - \> update: precip_nc_file = './IMERG_20230101_+7.nc'
    - \> execute: python3 NetCDF_to_CSV.py
