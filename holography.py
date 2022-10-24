import re
import tqdm
import os
import zarr
import numpy
import xarray
import cngi.conversion as conversion
import cngi.dio as dio
import numba as nb

from numba import types
from numba.typed import Dict
from time import perf_counter

def _convert_from_datetime(time:numpy.ndarray):
    """_summary_

    Args:
        time (numpy.ndarray): _description_

    Returns:
        _type_: _description_
    """

    return time.astype('float64')*(1e-9) + 3506716800.0

#@nb.njit(cache=True)
def _get_nearest_time_value(value:numpy.ndarray, times:xarray.core.dataset.Dataset)->numpy.ndarray:
    """_summary_

    Args:
        value (numpy.ndarray): _description_
        times (xarray.core.dataset.Dataset): _description_

    Returns:
        numpy.ndarray: _description_
    """
    index = numpy.abs(times - value).argmin()
    return index, times[index]


def _get_spw_list(msdx: xarray.core.dataset.Dataset)->list:
    """_summary_

    Args:
        msdx (xarray.core.dataset.Dataset): _description_

    Returns:
        list: _description_
    """
    spw = []
    for key in msdx.attrs.keys():
        match = re.match(r"xds[0-9]+", key)
        if match is not None:
            spw.append(match[0])

    return spw


def _compute_antenna_combinations(antenna1: numpy.ndarray, antenna2: numpy.ndarray) -> numpy.ndarray:
    """_summary_

    Args:
        antenna1 (numpy.ndarray): _description_
        antenna2 (numpy.ndarray): _description_

    Returns:
        numpy.ndarray: _description_
    """

    baselines = numpy.array([]) # amkethis preallocated in size
    for left in antenna1:
        antennas = antenna2[antenna2 > left]
        for right in antennas:
            baselines = numpy.append(baselines, [left, right])

    return baselines.reshape(baselines.shape[0] // 2, 2)


def _make_baseline_to_antenna_map(msdx: xarray.core.dataset.Dataset) -> dict:
    """_summary_

    Args:
        msdx (xarray.core.dataset.Dataset): _description_

    Returns:
        dict: _description_
    """

    antenna1_list = msdx.ANTENNA1.data.compute()
    antenna2_list = msdx.ANTENNA2.data.compute()

    #baseline_map = {}
    

    baseline_map = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )

    for i, antenna_tuple in enumerate(zip(antenna1_list, antenna2_list)):
        baseline_map[i] = numpy.asarray(antenna_tuple)

    return baseline_map

def _filter_pointing_table(pointing:xarray.core.dataarray.DataArray, antenna_id:int)->xarray.core.dataarray.DataArray:
    """_summary_

    Args:
        pointing (xarray.core.dataset.Dataset): _description_
        antenna_id (int): _description_

    Returns:
        xarray.core.dataset.Dataset: _description_
    """

    # Create direction subtable for a given antenna_id.
    subtable = pointing.DIRECTION.sel(antenna_id=antenna_id)

    # Build a mask on from the condition of any NaN values along axis=1.
    mask = ~numpy.isnan(subtable.data).any(axis=1)
    
    # Get the indicies from the mask that hold non-NaN data. The shape of the index
    # is (row/time=n_time, direction_tuple=2). The values along the row/time axis are 
    # doubled so we just get every other entry.
    index = numpy.where(mask.compute()==True)[0][::2]

    # Build new subtable containg properly time indexed direction values that are
    # not NaN. This time we use the .isel() command because we wnat hte index not 
    # the value to serve as the filter.

    return subtable.isel(time=index)


def _build_pointing_table_filter_indicies(pointing:xarray.core.dataarray.DataArray, antenna_list:list)->dict:
    """_summary_

    Args:
        pointing (xarray.core.dataarray.DataArray): _description_
        antenna_list (list): _description_

    Returns:
        dict: _description_
    """
    index_dict = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )

    for antenna in antenna_list:
        # Create direction subtable for a given antenna_id.
        subtable = pointing.DIRECTION.sel(antenna_id=antenna)

        # Build a mask on from the condition of any NaN values along axis=1.
        mask = ~numpy.isnan(subtable.data).any(axis=1)
    
        # Get the indicies from the mask that hold non-NaN data. The shape of the index
        # is (row/time=n_time, direction_tuple=2). The values along the row/time axis are 
        # doubled so we just get every other entry.
        index_dict[antenna] = numpy.where(mask.compute()==True)[0][::2]

    return index_dict

#@nb.jit(nopython=True, cache=True)
def _select_by_antenna(data, device_list, device_id, type='value'):
    """_summary_

    Args:
        data (_type_): _description_
        device_list (_type_): _description_
        device_id (_type_): _description_
        type (str, optional): _description_. Defaults to 'value'.

    Returns:
        _type_: _description_
    """
    if type == 'value':
        for i, device in enumerate(device_list):
            if device == device_id:
                index = i
                
            else:
                pass
    else:
        index = device_id
        
    return data[:, index::4, ...]

#@nb.jit(nopython=True, cache=True)
def _select_by_time(data, time_list, value, type='value'):
    """_summary_

    Args:
        data (_type_): _description_
        device_list (_type_): _description_
        device_id (_type_): _description_
        type (str, optional): _description_. Defaults to 'value'.

    Returns:
        _type_: _description_
    """
    #if type == 'value':
    #    for i, time in enumerate(time_list):
    #        if time == value:
    #            index = i
    #        else:
    #            pass
    #else:
    #    index = value
    index = value

    return data[index, ...]

#@nb.jit(nopython=True, cache=True)
def _build_pointing_table(direction, data, time_centroid, map, antenna_list, time_list, time_list_numerical, pointing_index_dict, n_time_centroids, n_baselines)->numpy.ndarray:
  
    #for time in tqdm.trange(n_time_centroids):
    #    for baseline in range(n_baselines):
    for time in range(2):
        for baseline in range(2):
            antennas = map[baseline]
            
            for n, antenna in enumerate(antennas):          
                
                subtable = pointing.sel(antenna_id=antenna).isel(time=pointing_index_dict[antenna])
                
                pointing_datetimes = subtable.coords['time'].data
                
                pointing_times = _convert_from_datetime(pointing_datetimes)
                                                
                index, nearest_time = _get_nearest_time_value(time_centroid[time, baseline], pointing_times)             
                
                datetime_index = numpy.where(numpy.datetime64(pointing_datetimes[index]) == pointing_datetimes)[0]
                
                direction = subtable.isel(time=datetime_index).DIRECTION.data.compute()

                direction = numpy.squeeze(direction)

                data[time, baseline, 0, RA] = direction[RA]
                data[time, baseline, 0, DEC] = direction[DEC]

            
    
    return data
    


if __name__ == '__main__':
    
    N_ANTENNAS = 2
    N_DIRECTIONS = 2
    RA = 0
    DEC = 1

    ignore = ['CALDEVICE', 'FEED', 'FIELD',
              'FLAG_CMD', 'HISTORY', 'OBSERVATION',
              'POLARIZATION', 'SOURCE',
              'STATE', 'SYSCAL', 'SYSPOWER', 'WEATHER', 'OVER_THE_TOP']

    if os.path.exists('evla_bctest.vis.zarr'):
        msdx = dio.read_vis(infile='evla_bctest.vis.zarr')

    else:
        msdx = conversion.convert_ms(infile='evla_bctest.ms') 

    spectral_windows = _get_spw_list(msdx)
    pointing = msdx.POINTING
    pointing_direction_array = pointing.DIRECTION.data.compute()
    
    start = perf_counter()

    for spw in tqdm.tqdm(spectral_windows, desc='processing spectral windows ...'):
        time_centroid = msdx.attrs[spw].TIME_CENTROID.data.compute()
        map = _make_baseline_to_antenna_map(msdx=msdx.attrs[spw])

        antenna_list = pointing.coords['antenna_id'].data
        time_list = pointing.coords['time'].data
        
        print('filtering ...')
        pointing_index_dict = _build_pointing_table_filter_indicies(pointing, antenna_list)

        n_time_centroids, n_baselines = msdx.attrs[spw].TIME_CENTROID.shape

        data = numpy.zeros([n_time_centroids, n_baselines, N_ANTENNAS, N_DIRECTIONS])

        pointing_datetimes = numpy.take(time_list, pointing_index_dict[0])
        time_list_numerical = _convert_from_datetime(time_list)

        print('building ...')
        data = _build_pointing_table(
            pointing_direction_array, 
            data, 
            time_centroid, 
            map, antenna_list, 
            time_list, 
            time_list_numerical, 
            pointing_index_dict, 
            n_time_centroids, 
            n_baselines
        )
                    
        xr = xarray.DataArray(
            data, 
            dims=['time', 'baseline', 'antenna', 'direction'], 
            coords=[time_centroid[:, 0], msdx.attrs[spw].coords['baseline'].data, [0, 1], [RA, DEC]]
        )
        
        zarr.save('evla_bctest_spw-{}.zarr'.format(spw), xr.to_numpy())
    stop = perf_counter()
    print('Total computation time ... {}'.format(stop - start))
