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

@nb.jit(nopython=True, cache=True)
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

@nb.jit(nopython=True, cache=True)
def _select_by_antenna(data, device_list, device_id, type='value'):
    if type == 'value':
        index = numpy.where(device_list==device_id)[0][0]
        
    else:
        index = device_id
        
    return data[:, index::4, ...][:, :, 0, :]


@nb.jit(nopython=True, cache=True)
def _build_pointing_table(direction_data, data, time_centroid, antenna_map, antenna_list, pointing_times, pointing_index_dict, n_time_centroids, n_baselines)->numpy.ndarray:
    print('Time iteration: ')
    for time in range(n_time_centroids):
        print(time)
        for baseline in range(n_baselines):
            antennas = antenna_map[baseline]
            
            for n, antenna in enumerate(antennas):     
                # Find the antenna_id index     
                index = numpy.where(antenna_list==antenna)[0][0]

                # Get the subtable for the given antenna_id and get rid of
                # the additinoal dimension
                c_subtable = direction_data[:, index::4, ...][:, :, 0, :]

                # Use the filter to get only the time valu entries that are
                # no NaN from the subtable.
                c_subtable = c_subtable[pointing_index_dict[antenna]]

                # Get the index of the nearest pointing time compared to the
                # time centroid value.
                index, _ = _get_nearest_time_value(
                    time_centroid[time, antenna], 
                    pointing_times[pointing_index_dict[antenna]]
                )

                # Get the direction (RA, DEC) values for the relevant time.
                direction = c_subtable[index, ...][0]

                # Write values to the new pointing table.
                data[time, baseline, n, 0] = direction[0]
                data[time, baseline, n, 1] = direction[1]            
    
    return data
    

def build_pointing_table(infile=''):
    
    N_ANTENNAS = 2
    N_DIRECTIONS = 2
    RA = 0
    DEC = 1

    ignore = ['CALDEVICE', 'FEED', 'FIELD',
              'FLAG_CMD', 'HISTORY', 'OBSERVATION',
              'POLARIZATION', 'SOURCE',
              'STATE', 'SYSCAL', 'SYSPOWER', 'WEATHER', 'OVER_THE_TOP']

    infile = 'group1.ms'

    if os.path.exists(infile.split('.')[0] + '.vis.zarr'):
        msdx = dio.read_vis(infile=infile.split('.')[0] + '.vis.zarr')

    else:
        msdx = conversion.convert_ms(infile=infile) 

    spectral_windows = _get_spw_list(msdx)
    pointing = msdx.POINTING
    pointing_direction_array = pointing.DIRECTION.data.compute()
    
    start = perf_counter()

    for spw in tqdm.tqdm(spectral_windows, desc='processing spectral windows ...'):
    #windows = ['xds7']
    #for spw in tqdm.tqdm(windows, desc='processing spectral windows ...'):
        time_centroid = msdx.attrs[spw].TIME_CENTROID.data.compute()
        antenna_map = _make_baseline_to_antenna_map(msdx=msdx.attrs[spw])

        antenna_list = pointing.coords['antenna_id'].data
        time_list = pointing.coords['time'].data
        
        pointing_index_dict = _build_pointing_table_filter_indicies(pointing, antenna_list)

        n_time_centroids, n_baselines = msdx.attrs[spw].TIME_CENTROID.shape

        data = numpy.zeros([n_time_centroids, n_baselines, N_ANTENNAS, N_DIRECTIONS])
        pointing_times = _convert_from_datetime(time_list)

        print('building pointing table ...')
        data = _build_pointing_table(
            pointing_direction_array, 
            data, 
            time_centroid, 
            antenna_map, 
            antenna_list, 
            pointing_times, 
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
