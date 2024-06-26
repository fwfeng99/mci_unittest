import os
import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u


def work_sext(
    image_path: str,
    config_path: str,
    out_path: str,
    flag_path: str,
    weight_path: str,
    out_para_path: str,
    filter_path: str,
):
    """
    Do astrometry by source-extractor.

    Do astrometry by source-extractor, the basic use of source-extractor is 'source-extractor  -c ' + fp_config + fp_image'

    Parameters
    ----------
    image_path : str
        The image fits to be processed, *.fits.
    config_path : str
        The config file of source-extractor, *.sex.
    out_path : str
        The name of the output catalog.
    out_para_path : str
        The parameters of the output catalog, *.param.
    filter_path : str
        The filter of the image, *.conv.

    Returns
    -------
    None

    Examples
    -------
    >>> work_sext(fp_img,fp_config_sext,fp_out_sext,fp_para_sext,fp_filter_sext)
    """
    fp_image = image_path
    fp_config = config_path
    fp_out = out_path
    # fp_flg = flag_path
    # fp_wht = weight_path
    fp_para = out_para_path
    fp_filter = filter_path

    cmd_sex = (
        "sex -c "
        + fp_config
        + " "
        + fp_image
        + " -CATALOG_NAME "
        + fp_out
        + " -FLAG_IMAGE "
        # + fp_flg
        + " -WEIGHT_IMAGE "
        # + fp_wht
        + " -PARAMETERS_NAME "
        + fp_para
        + " -FILTER_NAME "
        + fp_filter
    )

    os.system(cmd_sex)


def gaiadr3_query(
    ra: list,
    dec: list,
    rad: float = 1.0,
    maxmag: float = 25,
    maxsources: float = 1000000,
):
    """
    Acquire the Gaia DR3.

    This function uses astroquery.vizier to query Gaia DR3 catalog.

    Parameters
    ----------
    ra : list
        RA of center in degrees.
    dec : list
        Dec of center in degrees.
    rad : float
        Field radius in degrees.
    maxmag : float
        Upper limit magnitude.
    maxsources : float
        Maximum number of sources.

    Returns
    -------
    Table
        table of reference catalog.

    Examples
    --------
    >>> catalog = gaiadr3_query(ra, dec, rad, maxmag, maxsources)
    """

    vquery = Vizier(
        columns=["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Plx", "RVDR2", "Gmag", "RV"],
        row_limit=maxsources,
        column_filters={"Gmag": ("<%f" % maxmag), "Plx": ">0"},
    )
    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame="icrs")
    r = vquery.query_region(coord, radius=rad * u.deg, catalog="I/355/gaiadr3")

    return r[0]


def ideal_cel_coord(xi, eta, ra_c, dec_c):
    ra_c = np.deg2rad(ra_c)
    dec_c = np.deg2rad(dec_c)

    ra = np.arctan(xi / (np.cos(dec_c) - eta * np.sin(dec_c))) + ra_c
    dec = np.arctan(
        (eta * np.cos(dec_c) + np.sin(dec_c))
        / (np.cos(dec_c) - eta * np.sin(dec_c))
        * np.cos(ra - ra_c)
    )
    ra = np.degrees(ra)
    dec = np.degrees(dec)

    return ra, dec


def cel_ideal_coord(ra, dec, ra_c, dec_c):
    ra = np.radians(ra)
    dec = np.radians(dec)
    ra_c = np.radians(ra_c)
    dec_c = np.radians(dec_c)

    xi = (np.cos(dec) * np.sin(ra - ra_c)) / (
        np.sin(dec_c) * np.sin(dec) + np.cos(dec_c) * np.cos(dec) * np.cos(ra - ra_c)
    )
    eta = (
        np.cos(dec_c) * np.sin(dec) - np.sin(dec_c) * np.cos(dec) * np.cos(ra - ra_c)
    ) / (np.sin(dec_c) * np.sin(dec) + np.cos(dec_c) * np.cos(dec) * np.cos(ra - ra_c))

    return xi, eta


def pos_side(ra_a, dec_a, ra_b, dec_b):
    ra_a_c = np.deg2rad(ra_a)
    dec_a_c = np.deg2rad(dec_a)
    ra_b_c = np.deg2rad(ra_b)
    dec_b_c = np.deg2rad(dec_b)

    a_vector = np.array(
        [
            np.cos(dec_a_c) * np.cos(ra_a_c),
            np.cos(dec_a_c) * np.sin(ra_a_c),
            np.sin(dec_a_c),
        ]
    )
    b_vector = np.array(
        [
            np.cos(dec_b_c) * np.cos(ra_b_c),
            np.cos(dec_b_c) * np.sin(ra_b_c),
            np.sin(dec_b_c),
        ]
    )
    return np.rad2deg(np.arccos(np.sum(a_vector * b_vector, axis=0)))


def cal_plate_model(X_x, X_y, Y_xi, Y_eta, cof):
    X_xy = []
    for i in range(0, cof + 1):
        for j in range(0, i + 1):
            X_xy.append(X_x ** (i - j) * X_y**j)
    X_xy = np.array(X_xy).T

    A_CD1 = np.linalg.inv(X_xy.T @ X_xy) @ X_xy.T @ Y_xi
    A_CD2 = np.linalg.inv(X_xy.T @ X_xy) @ X_xy.T @ Y_eta

    A_CD = np.array([A_CD1, A_CD2])

    return A_CD


def use_plate_model(X_x, X_y, A_CD, cof):
    X_xy = []
    for i in range(0, cof + 1):
        for j in range(0, i + 1):
            X_xy.append(X_x ** (i - j) * X_y**j)
    X_xy = np.array(X_xy).T

    xi_eta = A_CD @ X_xy.T

    return xi_eta


def cal_plate_model_mag(X_x, X_y, mag, Y_xi, Y_eta, cof, cof_mag):
    X_xy = []
    for i in range(0, cof + 1):
        for j in range(0, i + 1):
            X_xy.append(X_x ** (i - j) * X_y**j)
    if cof_mag == 0:
        pass
    else:
        for k in range(1, cof_mag + 1):
            X_xy.append(mag**k)
    # X_xy.append(mag * X_x)
    # X_xy.append(mag * X_y)
    # X_xy.append(mag * X_x**2)
    # X_xy.append(mag * X_x * X_y)
    # X_xy.append(mag * X_y**2)
    # X_xy.append(mag * X_x, mag * X_y, mag * X_x ** 2, mag * X_x * X_y, mag * X_y ** 2)

    X_xy = np.array(X_xy).T

    A_CD1 = np.linalg.inv(X_xy.T @ X_xy) @ X_xy.T @ Y_xi
    A_CD2 = np.linalg.inv(X_xy.T @ X_xy) @ X_xy.T @ Y_eta

    A_CD = np.array([A_CD1, A_CD2])

    return A_CD


def use_plate_model_mag(X_x, X_y, mag, A_CD, cof, cof_mag):
    X_xy = []
    for i in range(0, cof + 1):
        for j in range(0, i + 1):
            X_xy.append(X_x ** (i - j) * X_y**j)

    if cof_mag == 0:
        pass
    else:
        for k in range(1, cof_mag + 1):
            X_xy.append(mag**k)
    # X_xy.append(mag * X_x)
    # X_xy.append(mag * X_y)
    # X_xy.append(mag * X_x**2)
    # X_xy.append(mag * X_x * X_y)
    # X_xy.append(mag * X_y**2)
    # X_xy.append(mag * X_x, mag * X_y, mag * X_x ** 2, mag * X_x * X_y, mag * X_y ** 2)

    X_xy = np.array(X_xy).T

    xi_eta = A_CD @ X_xy.T

    return xi_eta


def cal_plate_model_(X_x, X_y, mag, Y_xi, Y_eta):
    X_xy_1 = np.array(
        [
            np.ones_like(X_x),
            X_x,
            X_y,
            X_x**2,
            X_x * X_y,
            X_y**2,
            X_x**3,
            X_x**2 * X_y,
            X_x * X_y**2,
            X_y**3,
            mag,
            mag**2,
            mag**3,
            mag**4,
            mag * X_x,
            mag * X_y,
        ]
    ).T
    X_xy_2 = np.array(
        [
            np.ones_like(X_x),
            X_x,
            X_y,
            X_x**2,
            X_x * X_y,
            X_y**2,
            X_x**3,
            X_x**2 * X_y,
            X_x * X_y**2,
            X_y**3,
            mag,
            mag**2,
            mag**3,
            mag**4,
            mag * X_x,
            mag * X_y,
        ]
    ).T
    # X_xy = np.array([np.ones_like(X_x), X_x, X_y, X_x ** 2, X_x * X_y, X_y ** 2]).T
    # X_xy_1 = np.array([np.ones_like(X_x), X_x, X_y, X_x ** 2, X_x * X_y, X_y ** 2]).T

    A_CD1 = np.linalg.inv(X_xy_1.T @ X_xy_1) @ X_xy_1.T @ Y_xi
    A_CD2 = np.linalg.inv(X_xy_2.T @ X_xy_2) @ X_xy_2.T @ Y_eta
    A_CD = np.array([A_CD1, A_CD2])

    return A_CD


def use_plate_model_(X_x, X_y, mag, A_CD):
    X_xy_1 = np.array(
        [
            np.ones_like(X_x),
            X_x,
            X_y,
            X_x**2,
            X_x * X_y,
            X_y**2,
            X_x**3,
            X_x**2 * X_y,
            X_x * X_y**2,
            X_y**3,
            mag,
            mag**2,
            mag**3,
            mag**4,
            mag * X_x,
            mag * X_y,
        ]
    ).T
    X_xy_2 = np.array(
        [
            np.ones_like(X_x),
            X_x,
            X_y,
            X_x**2,
            X_x * X_y,
            X_y**2,
            X_x**3,
            X_x**2 * X_y,
            X_x * X_y**2,
            X_y**3,
            mag,
            mag**2,
            mag**3,
            mag**4,
            mag * X_x,
            mag * X_y,
        ]
    ).T

    xi = A_CD[0] @ X_xy_1.T
    eta = A_CD[1] @ X_xy_2.T

    return np.array([xi, eta])


def write_ds9_region(filename, data, str_x, str_y):
    with open(filename, "w") as f:
        f.write(
            'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1"
            + "\n"
        )
        f.write("physical" + "\n")
        for i in range(len(data)):
            f.write(
                "circle("
                + str("%12.6f" % (data[str_x][i]))
                + ",  "
                + str("%12.6f" % (9232 - data[str_y][i]))
                + ","
                + str(15)
                + ")"
                + "\n"
            )


def match_box_id(points_x, points_y, bins_x, bins_y):
    # bins_x_id and bins_y_id
    bins_x_id = np.digitize(points_x, bins_x) - 1
    bins_y_id = np.digitize(points_y, bins_y) - 1

    return bins_x_id, bins_y_id


def cal_distort_model(
    dis_res_xy_list,
    bins_x,
    bins_y,
    bins_num,
    distort_sum_bins,
    distort_sum_values_x,
    distort_sum_values_y,
):
    for points_inf in dis_res_xy_list:
        points_x = points_inf[0, :]
        points_y = points_inf[1, :]
        distort_value_x = points_inf[2, :]
        distort_value_y = points_inf[3, :]

        bins_x_id, bins_y_id = match_box_id(points_x, points_y, bins_x, bins_y)

        distort_bins, _, _ = np.histogram2d(
            bins_x_id, bins_y_id, bins=(range(bins_num + 1), range(bins_num + 1))
        )
        distort_values_x, _, _ = np.histogram2d(
            bins_x_id,
            bins_y_id,
            bins=(range(bins_num + 1), range(bins_num + 1)),
            weights=distort_value_x,
        )
        distort_values_y, _, _ = np.histogram2d(
            bins_x_id,
            bins_y_id,
            bins=(range(bins_num + 1), range(bins_num + 1)),
            weights=distort_value_y,
        )

        distort_sum_bins += distort_bins
        distort_sum_values_x += distort_values_x
        distort_sum_values_y += distort_values_y

    return distort_sum_bins, distort_sum_values_x, distort_sum_values_y


def sigma_filter(data, data_rel, sigma_scale):
    mean_data = np.mean(data)
    std_data = np.std(data)
    # 小于μ-3σ或大于μ+3σ的数据均为异常值
    filter = (mean_data - sigma_scale * std_data < data) & (
        mean_data + sigma_scale * std_data > data
    )

    data_filter = data[filter]
    data_rel_filter = data_rel[filter]

    return data_filter, data_rel_filter


def sigma_filter_2(data1, data2, data_rel, sigma_scale):
    mean_data1 = np.mean(data1)
    std_data1 = np.std(data1)

    mean_data2 = np.mean(data2)
    std_data2 = np.std(data2)
    # 小于μ-3σ或大于μ+3σ的数据均为异常值
    filter1 = (mean_data1 - sigma_scale * std_data1 < data1) & (
        mean_data1 + sigma_scale * std_data1 > data1
    )
    filter2 = (mean_data2 - sigma_scale * std_data2 < data2) & (
        mean_data2 + sigma_scale * std_data2 > data2
    )

    filter = filter1 & filter2

    data1_filter = data1[filter]
    data2_filter = data2[filter]
    data_rel_filter = data_rel[filter]

    return data1_filter, data2_filter, data_rel_filter, filter
