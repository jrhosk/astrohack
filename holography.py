import re
import numpy as np
import xarray as xr
import cngi.conversion as conversion


def _get_spw_list(msdx: xr.core.dataset.Dataset):
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


def _compute_antenna_combinations(antenna1: np.ndarray, antenna2: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    antenna1
    antenna2

    Returns
    -------

    """

    baselines = np.array([])
    for left in antenna1:
        antennas = antenna2[antenna2 > left]
        for right in antennas:
            baselines = np.append(baselines, [left, right])

    return baselines.reshape(baselines.shape[0] // 2, 2)


def _make_baseline_to_antenna_map(msdx: xr.core.dataset.Dataset) -> dict:
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
    ignore = ['CALDEVICE', 'FEED', 'FIELD',
              'FLAG_CMD', 'HISTORY', 'OBSERVATION',
              'POLARIZATION', 'SOURCE',
              'STATE', 'SYSCAL', 'SYSPOWER', 'WEATHER']

    msdx = conversion.convert_ms(infile='evla_bctest.ms', ignore=ignore)

    map = _make_baseline_to_antenna_map(msdx=msdx.xds0)
    spectral_windows = _get_spw_list(msdx)

    print(spectral_windows)
