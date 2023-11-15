import netCDF4
import pandas as pd
import numpy as np

precip_nc_file = './IMERG_20230101_+7.nc'
nc = netCDF4.Dataset(precip_nc_file, mode='r')

lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
precip = nc.variables['precipitationCal'][:]

# Create a meshgrid of latitude and longitude coordinates
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Flatten the grids and precipitation data
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()
precip_flat = precip.flatten()

# Create a multi-index from latitudes and longitudes
multi_index = pd.MultiIndex.from_arrays([lat_flat, lon_flat], names=('lat', 'lon'))

# Create a DataFrame with the multi-index
df = pd.DataFrame({'precipitation': precip_flat}, index=multi_index)

# Save to CSV
df.to_csv('precip.csv')
