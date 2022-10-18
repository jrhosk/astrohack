import re
import tqdm
import zarr
import numpy
import xarray
import cngi.conversion as conversion
import cngi.dio as dio

def _convert_from_datetime(time:numpy.ndarray):
    """
    Some _summary_
    """

    return time.astype('float64')*(1e-9) + 3506716800.0

def _get_nearest_time_value(value:numpy.ndarray, times:xarray.core.dataset.Dataset)->numpy.ndarray:
    """

    Args:
        value (numpy.ndarray): _description_
        times (xarray.core.dataset.Dataset): _description_
    """
    index = numpy.abs(times - value).argmin()
    return index, times[index]


def _get_spw_list(msdx: xarray.core.dataset.Dataset)->list:
    """

    Parameters
    ----------
    msdx: xarray dataset containing measurement table

    Returns: List of spectral windows contained in measurement table
    -------

    """
    spw = []
    for key in msdx.attrs.keys():
        match = re.match(r"xds[0-9]+", key)
        if match is not None:
            spw.append(match[0])

    return spw


def _compute_antenna_combinations(antenna1: numpy.ndarray, antenna2: numpy.ndarray) -> numpy.ndarray:
    """

    Parameters
    ----------
    antenna1
    antenna2

    Returns
    -------

    """

    baselines = numpy.array([])
    for left in antenna1:
        antennas = antenna2[antenna2 > left]
        for right in antennas:
            baselines = numpy.append(baselines, [left, right])

    return baselines.reshape(baselines.shape[0] // 2, 2)


def _make_baseline_to_antenna_map(msdx: xarray.core.dataset.Dataset) -> dict:
    """

    Parameters
    ----------
    msdx

    Returns
    -------

    """

    antenna1_list = msdx.ANTENNA1.data.compute()
    antenna2_list = msdx.ANTENNA2.data.compute()

    baseline_map = {}

    for i, antenna_tuple in enumerate(zip(antenna1_list, antenna2_list)):
        baseline_map[i] = antenna_tuple

    return baseline_map


if __name__ == '__main__':
    
    N_ANTENNAS = 2
    N_DIRECTIONS = 2
    RA = 0
    DEC = 1

    ignore = ['CALDEVICE', 'FEED', 'FIELD',
              'FLAG_CMD', 'HISTORY', 'OBSERVATION',
              'POLARIZATION', 'SOURCE',
              'STATE', 'SYSCAL', 'SYSPOWER', 'WEATHER', 'OVER_THE_TOP']

    
    msdx = dio.read_vis(infile='evla_bctest.vis.zarr')

    #msdx = conversion.convert_ms(infile='evla_bctest.ms', ignore=ignore)
    pointing = msdx.POINTING
    
    scan_times = _convert_from_datetime(pointing.coords['time'].data)
    
    spectral_windows = _get_spw_list(msdx)

    for spw in tqdm.tqdm(spectral_windows, desc='processing spectral windows ...'):
        time_centroid = msdx.attrs[spw].TIME_CENTROID.data.compute()
        map = _make_baseline_to_antenna_map(msdx=msdx.attrs[spw])

        n_time_centroids, n_baselines = msdx.attrs[spw].TIME_CENTROID.shape

        data = numpy.zeros([n_time_centroids, n_baselines, N_ANTENNAS, N_DIRECTIONS])

        for time in tqdm.trange(n_time_centroids, desc='processing time, baseline ...'):
            for baseline in range(n_baselines):
                antennas = map[baseline]

                for n, antenna in enumerate(antennas):            
                    pointing_datetimes = pointing.isel(antenna_id=antenna).coords['time'].data
            
                    pointing_times = _convert_from_datetime(pointing_datetimes)

                    index, nearest_time = _get_nearest_time_value(time_centroid[time, baseline], pointing_times)
                    datetime_index = numpy.where(numpy.datetime64(pointing_datetimes[index]) == pointing_datetimes)[0]
                    direction = pointing.isel(antenna_id=antenna, time=datetime_index).DIRECTION.data.compute()
                    direction = numpy.squeeze(direction)

                    data[time, baseline, n, RA] = direction[RA]
                    data[time, baseline, n, DEC] = direction[DEC]
    
        xr = xarray.DataArray(
            data, 
            dims=['time', 'baseline', 'antenna', 'direction'], 
            coords=[time_centroid[:, 0], msdx.attrs[spw].coords['baseline'].data, [0, 1], [RA, DEC]]
        )
        
        zarr.save('evla_bctest_spw-{}.zarr'.format(spw), xr.to_numpy())
