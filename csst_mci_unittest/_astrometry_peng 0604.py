"""
Identifier:     csst_mci_astrometry/astrometry.py
Name:           astrometry.py
Description:    astrometry code for MCI
Author:         Xiyan Peng
Created:        2023-11-25
Modified-History:
    2023-11-14, Xiyan Peng, created
    2023-12-29, Xiyan Peng, modified
    2023-12-11, Xiyan Peng, add the docstring
    2024-01-07, Xiyan Peng, add the docstring and update function name
"""
import os
import numpy as np

from astropy import units as u
from astropy.table import Table, Column
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, Distance, CartesianRepresentation
from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import sigma_clip
from astroquery.vizier import Vizier

from csst_mci_common.logger import get_logger
from csst_mci_common.status import CsstStatus, CsstResult
from csst_mci_common.data_manager import CsstMCIDataManager
import joblib
import shutil




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
        columns=["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Plx", "RVDR2", "Gmag"],
        row_limit=maxsources,
        column_filters={"Gmag": ("<%f" % maxmag), "Plx": ">0"},
    )
    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame="icrs")
    r = vquery.query_region(coord, radius=rad * u.deg, catalog="I/355/gaiadr3")

    return r[0]


def update_catalog_space_motion(
    catalog: Table,
    header: fits.header.Header,
    dfsapi: bool = False
):
    """
    Update reference catalog.

    Update reference catalog to observed epoch by proper motion, parallax.

    Parameters
    ----------
    catalog : Table
        Reference catalog table.
    hearder : fits.header.Header
        The header of image fits.
    dfsapi : bool
        Whether use dfs data.

    Returns
    -------
    Table
        table of update reference catalog.

    Examples
    --------
    >>> catalog = gaiadr3_query(ra, dec, rad, maxmag, maxsources)
    """
    if dfsapi:
        ra = catalog["Ra"]  # For DFS api
        dec = catalog["Dec"]
        parallax = catalog["Parallax"]
        pmra = catalog["Pmra"]  # .value
        pmdec = catalog["Pmdec"]  # .value
        mag = catalog["Gmag"]
    else:
        ra = catalog["RA_ICRS"]  # .value
        dec = catalog["DE_ICRS"]  # .value
        parallax = catalog["Plx"]  # .value
        pmra = catalog["pmRA"]  # .value
        pmdec = catalog["pmDE"]  # .value
        mag = catalog["Gmag"]

    dt2 = TimeDelta(header["EXPTIME"], format="sec")
    # daytime = header['DATE-OBS'] + 'T' + header['TIME-OBS']
    daytime = header["DATE-OBS"]
    t = Time(daytime, format="isot", scale="utc")
    epoch_now = t  # + dt2
    input_x = header["POSI0_X"]
    input_y = header["POSI0_Y"]
    input_z = header["POSI0_Z"]
    input_vx = header["VELO0_X"]
    input_vy = header["VELO0_Y"]
    input_vz = header["VELO0_Z"]
    tt1 = CartesianRepresentation(
        input_x * 1000 * u.m, input_y * 1000 * u.m, input_z * 1000 * u.m
    )

    tt2 = CartesianRepresentation(
        input_vx * 1000 * u.m / u.s,
        input_vy * 1000 * u.m / u.s,
        input_vz * 1000 * u.m / u.s,
    )

    cut = (abs(pmra) > 0) & (abs(pmdec) > 0)
    catpm = catalog[cut]

    c = SkyCoord(
        ra=ra[cut],
        dec=dec[cut],
        distance=Distance(parallax=abs(parallax[cut]) * u.mas),
        pm_ra_cosdec=pmra[cut],
        pm_dec=pmdec[cut],
        obstime=Time(2016.0, format="jyear", scale="utc"),
        frame="icrs",
    )

    epochobs = Time(2000.0, format="jyear", scale="utc")
    c_epoch_now = c.apply_space_motion(epochobs)

    catalog_now = Table()
    catalog_now.add_column(Column(name="X_WORLD", data=c_epoch_now.ra.degree, unit=u.deg))
    catalog_now.add_column(Column(name="Y_WORLD", data=c_epoch_now.dec.degree, unit=u.deg))
    catalog_now.add_column(Column(name="pmRA", data=c_epoch_now.pm_ra_cosdec, unit=u.mas / u.yr))
    catalog_now.add_column(Column(name="pmDE", data=c_epoch_now.pm_dec, unit=u.mas / u.yr))
    catalog_now.add_column(Column(name="Plx",data=c_epoch_now.distance.to(u.arcsec, equivalencies=u.parallax()),unit=u.mas,))
    catalog_now.add_column(Column(name="MAG", data=mag[cut], unit=u.mag))
    catalog_now.add_column(Column(name="ERRA_WORLD", data=mag[cut] / 1000 / 3600))
    catalog_now.add_column(Column(name="ERRB_WORLD", data=mag[cut] / 1000 / 3600))

    return catalog_now


def convert_hdu_to_ldac(hdu: fits.hdu.hdulist.HDUList):
    """
    Convert hdu table to ldac format.

    Convert an hdu table to a fits_ldac table (format used by astromatic suite).

    Parameters
    ----------
    hdu : fits.hdu.hdulist.HDUList
        HDUList to convert to fits_ldac HDUList.

    Returns
    -------
    tuple:
        The tuple contains Header info for fits table (LDAC_IMHEAD) and Data table (LDAC_OBJECTS),
            the type of table is fits.BinTableHDU.

    Examples
    --------
    >>> tbl = convert_hdu_to_ldac(hdu)
    """

    tblhdr = np.array([hdu[1].header.tostring()])
    hcol = fits.ColDefs([fits.Column(name="Field Header Card", array=tblhdr, format="13200A")])
    tbl1 = fits.BinTableHDU.from_columns(hcol)
    tbl1.header["TDIM1"] = "(80,   {0})".format(len(hdu[1].header))
    tbl1.header["EXTNAME"] = "LDAC_IMHEAD"

    dcol = fits.ColDefs(hdu[1].data)
    tbl2 = fits.BinTableHDU.from_columns(dcol)
    tbl2.header["EXTNAME"] = "LDAC_OBJECTS"
    return (tbl1, tbl2)



def initial_standard_output_head(image_path: str, head_output: str):
    """
    Write the WCS head.

    Write the new WCS head from *img.fits.

    Parameters
    ----------
    image_path : str
        The image to be processed, *img.fits.
    head_output : str
        A new head.
    Returns
    -------
    None

    Examples
    --------
    >>> initial_standard_output_head(fp_img, fp_head_ast)
    """

    if os.path.splitext(image_path)[-1] == ".fits":
        header = fits.getheader(image_path, 1)
        newheader = fits.Header()

        #        del newheader[3:22]
        newheader["STA_AST"] = (1, "Completion degree of MCI astrometric solution")
        newheader["VER_AST"] = ("v2023.01", "Version of MCI Astrometry soft")
        newheader["STM_AST"] = ("", "Time of last MCI Astrometry")

        newheader["EQUINOX"] = (2000.00000000, "Mean equinox")
        newheader["RADESYS"] = ("ICRS    ", "Astrometric system")
        newheader["CTYPE1"] = ("RA---TPV", "WCS projection type for this axis")
        newheader["CTYPE2"] = ("DEC--TPV", "WCS projection type for this axis")
        newheader["CUNIT1"] = ("deg     ", "Axis unit")
        newheader["CUNIT2"] = ("deg     ", "Axis unit")
        newheader["CRVAL1"] = (header["CRVAL1"], "World coordinate on this axis")
        newheader["CRVAL2"] = (header["CRVAL2"], "World coordinate on this axis")
        newheader["CRPIX1"] = (header["CRPIX1"], "Reference pixel on this axis")
        newheader["CRPIX2"] = (header["CRPIX2"], "Reference pixel on this axis")

        newheader["CD1_1"] = (header["CD1_1"], "Linear projection matrix")
        newheader["CD1_2"] = (header["CD1_2"], "Linear projection matrix")
        newheader["CD2_1"] = (header["CD2_1"], "Linear projection matrix")
        newheader["CD2_2"] = (header["CD2_2"], "Linear projection matrix")
        newheader["PV1_0"] = (0, "Projection distortion parameter")
        newheader["PV1_1"] = (1, "Projection distortion parameter")
        newheader["PV1_2"] = (0, "Projection distortion parameter")
        newheader["PV1_4"] = (0, "Projection distortion parameter")
        newheader["PV1_5"] = (0, "Projection distortion parameter")
        newheader["PV1_6"] = (0, "Projection distortion parameter")
        newheader["PV1_7"] = (0, "Projection distortion parameter")
        newheader["PV1_8"] = (0, "Projection distortion parameter")
        newheader["PV1_9"] = (0, "Projection distortion parameter")
        newheader["PV1_10"] = (0, "Projection distortion parameter")
        newheader["PV1_12"] = (0, "Projection distortion parameter")
        newheader["PV1_13"] = (0, "Projection distortion parameter")
        newheader["PV1_14"] = (0, "Projection distortion parameter")
        newheader["PV1_15"] = (0, "Projection distortion parameter")
        newheader["PV1_16"] = (0, "Projection distortion parameter")
        newheader["PV2_0"] = (0, "Projection distortion parameter")
        newheader["PV2_1"] = (1, "Projection distortion parameter")
        newheader["PV2_2"] = (0, "Projection distortion parameter")
        newheader["PV2_4"] = (0, "Projection distortion parameter")
        newheader["PV2_5"] = (0, "Projection distortion parameter")
        newheader["PV2_6"] = (0, "Projection distortion parameter")
        newheader["PV2_7"] = (0, "Projection distortion parameter")
        newheader["PV2_8"] = (0, "Projection distortion parameter")
        newheader["PV2_9"] = (0, "Projection distortion parameter")
        newheader["PV2_10"] = (0, "Projection distortion parameter")
        newheader["PV2_12"] = (0, "Projection distortion parameter")
        newheader["PV2_13"] = (0, "Projection distortion parameter")
        newheader["PV2_14"] = (0, "Projection distortion parameter")
        newheader["PV2_15"] = (0, "Projection distortion parameter")
        newheader["PV2_16"] = (0, "Projection distortion parameter")

        newheader["ASTIRMS1"] = (-9999, "Astrom. dispersion RMS (intern., high S/N)")
        newheader["ASTIRMS2"] = (-9999, "Astrom. dispersion RMS (intern., high S/N)")
        newheader["ASTRRMS1"] = (-9999, "Astrom. dispersion RMS (ref., high S/N)")
        newheader["ASTRRMS2"] = (-9999, "Astrom. dispersion RMS (ref., high S/N)")

        #        newheader['STA_CCRS'] = (
        #            1, "Completion degree of relative astrometric solution in MCI")
        #        newheader['VER_CCRS'] = (
        #            "v2023.01", "Version of CSST relative Astrometry soft in MCI")
        #        newheader['STM_CCRS'] = ("", "Time of last CSST Astrometry in MCI")

        newheader["ASTGATE"] = (" ", "Camera shutter information")
        newheader["ASTCONF"] = (" ", "Configuration file for astrometry")
        newheader["ASTSIM"] = ("normal", "Image classification for MCI Astrometry")
        newheader["GCRSREF"] = (
            "Gaia dr3 v01",
            "Reference Catalogue for MCI Astrometry",
        )
        newheader["ASTHIS"] = (1, "Astrometric solution Record for MCI Astrometry")
        newheader["DELT_RA"] = (-9999, "Change in central RA")
        newheader["DELT_dec"] = (-9999, "Change in central DEC")
        newheader["DELT_ps"] = (-9999, "Change in pixelscale")
        empty_primary = fits.PrimaryHDU(header=newheader)
        empty_primary.writeto(head_output, overwrite=True)


def work_sext(
    image_path: str,
    image_flag_path: str,
    image_weight_path: str,
    config_path: str,
    out_path: str,
    out_para_path: str,
    filter_path: str,
):
    """
    Do astrometry by Source-Extractor.

    Do astrometry by Source-Extractor, the basic use of Source-Extractor is 'source-extractor  -c ' + fp_config + fp_image'

    Parameters
    ----------
    image_path : str
        The image fits to be processed, *.fits.
    img_flag_path : str
        The flag image for the image, *.fits.
    img_weight_path : str
        The weight image for the image, *.fits.
    config_path : str
        The config file of Source-Extractor, *.sex.
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
    >>> work_sext(fp_img, fp_img_flag, fp_img_weight, fp_config_sext, fp_out_sext, fp_para_sext, fp_filter_sext)
    """
    fp_image = image_path
    fp_flag = image_flag_path
    fp_weight = image_weight_path
    fp_config = config_path
    fp_out = out_path
    fp_para = out_para_path
    fp_filter = filter_path

    cmd_sex = (
        "sex  -c "
        + fp_config
        + " "
        + fp_image
        + " -CATALOG_NAME "
        + fp_out
        + " -FLAG_IMAGE "
        + fp_flag
        + " -WEIGHT_IMAGE "
        + fp_weight
        + " -PARAMETERS_NAME "
        + fp_para
        + " -FILTER_NAME "
        + fp_filter
    )

    os.system(cmd_sex)




def update_sext_catalog(sext_out_path: str, sext_out_path_up: str):
    # Open star_info_SN_1.fits to get posX_C1 and posY_C1
    directory = os.path.dirname(sext_out_path)
    
    # Construct the path to star_info_SN_1.fits
    star_info_path = os.path.join(directory, "star_info_SN_1.fits")

    with fits.open(star_info_path) as star_info_hdulist:
        star_info_data = star_info_hdulist[1].data
        posX_C1 = star_info_data['posX_C1']
        posY_C1 =9232- star_info_data['posY_C1']
    
    fp_out = sext_out_path
    fp_out_up = sext_out_path_up

    # filter data from the output fits from Source-Extractor
    hdulist = fits.open(fp_out, mode="update")
    ptdata1 = hdulist[2].data  # LDAC
    ptdata1["FLUXERR_AUTO"] = ptdata1["FLUX_AUTO"] / 100

    # Match tdata with posX_C1 and posY_C1
    matched_indices = []
    for i in range(len(ptdata1)):
        distance = np.sqrt((ptdata1["X_IMAGE"][i] - posX_C1)**2 + (ptdata1["Y_IMAGE"][i] - posY_C1)**2)
        if np.min(distance) < 6:
            matched_indices.append(i)
    tdata = ptdata1[matched_indices]

    # Write positions of filtered data to file, *.reg.
    filesatreg = fp_out + ".reg"
    os.system("rm " + filesatreg)
    write_reg(tdata, filesatreg, "X_IMAGE", "Y_IMAGE")
    with open(filesatreg, "a") as f3:
        f3.write(
            'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1"
            + "\n"
        )
        f3.write("physical" + "\n")
        for kk in range(len(tdata)):
            f3.write(
                "circle("
                + str("%12.6f" % tdata["X_IMAGE"][kk])
                + ",  "
                + str("%12.6f" % tdata["Y_IMAGE"][kk])
                + ","
                + str(4)
                + ")"
                + "\n"
            )

    try:
        assert (len(tdata) > 30) and (len(tdata) < 30000)
    except AssertionError:
        raise AssertionError("too few stars or too much stars ")

    # Update the output fits from Source-Extractor, only change the data part (hdulist[2].data).
    hdulist[2].data = tdata
    hdulist.flush()
    hdulist.writeto(fp_out_up, overwrite=True)

    return tdata





def update_sext_catalog_old(sext_out_path: str, sext_out_path_up: str):
    """
    Update the fits of Source-Extractor.

    Update the fits of Source-Extractor, the part to be updated is data.

    Parameters
    ----------
    sext_out_path : str
        The result of Source-Extractor, the type is LDAC fits, however extension is .acat.
    sext_out_path_up : str
        The update fits, the type is LDAC fits, however extension is .acat.

    Returns
    -------
    tdata
        The updated data of the fits.

    Examples
    -------
    >>> stardata = update_sext_catalog(fp_out_sext, fp_out_sext_up)
    """
    fp_out = sext_out_path
    fp_out_up = sext_out_path_up

    # filter data from the output fits from Source-Extractor
    hdulist = fits.open(fp_out, mode="update")
    ptdata1 = hdulist[2].data  # LDAC
    ptdata1["FLUXERR_AUTO"] = ptdata1["FLUX_AUTO"] / 100
    tdata = ptdata1

    #region = (tdata["AWIN_IMAGE"] < 1.2) & (tdata["AWIN_IMAGE"] > 0.2)
    #tdata = tdata[region]
    #if len(tdata["ELLIPTICITY"]) > 800:
    #    temp = np.sort(tdata, order=["FLUX_AUTO"], axis=0)
    #    tdata = temp[-150:-1]

    # write positions of filtered data to file, *.reg.
    filesatreg = fp_out + ".reg"
    os.system("rm " + filesatreg)
    write_reg(tdata, filesatreg, "X_IMAGE", "Y_IMAGE")
    with open(filesatreg, "a") as f3:
         f3.write(
             'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" '
             "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1"
             + "\n"
         )
         f3.write("physical" + "\n")
         for kk in range(len(tdata)):
             f3.write(
                 "circle("
                 + str("%12.6f" % tdata["X_IMAGE"][kk])
                 + ",  "
                 + str("%12.6f" % tdata["Y_IMAGE"][kk])
                 + ","
                 + str(4)
                 + ")"
                 + "\n"
             )

    try:
        assert (len(tdata) > 30) and (len(tdata) < 30000)
    except AssertionError:
        raise AssertionError("too few stars or too much stars ")

    # update the output fits from Source-Extractor, only change the data part (hdulist[2].data).
    hdulist[2].data = tdata
    hdulist.flush()
    hdulist.writeto(fp_out_up, overwrite="True")

    return tdata


def write_reg(
    data,
    fp_reg: str,
    X_term: str,
    Y_term: str,
):
    """
    Write data to file, *.reg.

    Write positions of filtered data to file, *.reg.

    Parameters
    ----------
    data : fits.fitsrec.FITS_rec
        The filtered data of the Source-Extractor output.
    fp_reg : str
        The reg fits.
    X_term : str
        The X-axis name.
    Y_term : str
        The Y-axis name.

    Returns
    -------
    None

    Examples
    -------
    >>> write_reg(tdata, filesatreg, "X_IMAGE", "Y_IMAGE")
    """
    with open(fp_reg, "a") as f:
        f.write(
            'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1"
            + "\n"
        )
        f.write("physical" + "\n")
        for i in range(len(data)):
            f.write(
                "circle("
                + str("%12.6f" % data[X_term][i])
                + ",  "
                + str("%12.6f" % data[Y_term][i])
                + ","
                + str(4)
                + ")"
                + "\n"
            )


def work_scamp(
    fp_sext_out: str,
    fp_refcat: str,
    fp_config_para: str,
    fp_scamp_head: str,
    fp_up_head: str,
):
    """
    Do astrometry by Scamp.

    Do astrometry by Scamp from the result of Source-Extractor.

    Parameters
    ----------
    fp_sext_out : str
        Fits name of result of Source-Extractor.
    fp_refcat : str
        Gaia reference catalog.
    fp_config_para : str
        Scamp config file.
    fp_scamp_head : str
        Scamp result head file.
    fp_up_head : str
        The aheader suffix file.

    Returns
    -------
    None

    Examples
    -------
    >>> work_scamp(fp_sext_out, fp_refcat, fp_config_para, fp_scamp_head, fp_up_head)
    """

    cmd_scamp = (
        " scamp "
        + fp_sext_out
        + " -c "
        + fp_config_para
        + " -ASTREFCAT_NAME "
        + fp_refcat
        + " -AHEADER_SUFFIX "
        + fp_up_head
    )
    os.system(cmd_scamp)

    try:
        assert os.access(fp_scamp_head, os.F_OK)
    except AssertionError:
        raise AssertionError("no Scamp result head file")


def rewrite_scamp_head_to_standard(scamp_head_path: str, scamp_head_bk_path: str):
    """
    Rewrite the Scamp head.

    Rewrite the Scamp result head to the standard fits header, 2880 bytes for each block.

    Parameters
    ----------
    scamp_head_path : str
        Scamp head file name.
    scamp_head_bk_path : str
        Rewrited Scamp head file name.

    Returns
    -------
    None

    Examples
    --------
    >>> rewrite_scamp_head_to_standard(fp_head_scamp, fp_head_bk_scamp)
    """

    fp_head = scamp_head_path
    fp_head_bk = scamp_head_bk_path

    if os.path.splitext(fp_head)[-1] == ".head":
        f = open(fp_head, "r")
        f_bk = open(fp_head_bk, "w")
        a = ""
        i = 0
        for v in f.readlines():
            sp = ""
            asp = ""
            i += 1
            if len(v) <= 81:
                sp = " " * (81 - len(v))
            if "Sorbonne" in v:
                v = "COMMENT   (c) 2010-2018 Sorbonne Universite/Universite de Bordeaux/CNRS"
            if "END" in v:
                asp = " " * 80 * (36 - i % 36)
                i = i + (36 - i % 36)
                # print(i)
            a = a + v + sp + asp
        f_bk.write(a.replace("\n", ""))
        f_bk.close()
        f.close()


def check_astrometry_result(w: WCS, stardata: fits.fitsrec.FITS_rec, refcat_now: Table):
    """
    check_astrometry_result.

    check_astrometry_result.

    Parameters
    ----------
    w : WCS
        Scamp head file name.
    stardata : fits.fitsrec.FITS_rec
        The filtered data of the Source-Extractor output.
    refcat_now : Table
        Rewrited Scamp head file name.

    Returns
    -------
    meanra : float
        Mean of R.A.
    meande : float
        Mean of Declination.
    stdra : float
        Std of R.A.
    stdde : float
        Std of Declination.
    match_num : int
        Number of matched stars.

    Examples
    --------
    >>> check_astrometry_result(w, stardata, refcat_now)
    """

    coeff0 = coeff01 = 99999
    cstd = cstd1 = 99999
    match_num = 0
    obsc = w.pixel_to_world(stardata["XWIN_IMAGE"] - 1, stardata["YWIN_IMAGE"] - 1)
    mask = abs(refcat_now["X_WORLD"]) > 0  # and (abs( refdata['Y_WORLD'] >0 ))
    refdata = refcat_now[mask]
    gaia_ra = refdata["X_WORLD"]
    gaia_dec = refdata["Y_WORLD"]

    refc = SkyCoord(ra=gaia_ra, dec=gaia_dec)
    idx, d2d, d3d = obsc.match_to_catalog_sky(refc)
    ref_uid = np.unique(idx)
    obs_uid = np.full_like(ref_uid, -1)
    tmpj = -1
    ccdraoff_med = ccddecoff_med = ccdra_rms = ccddec_rms = -1
    for i in ref_uid:
        tmpj = tmpj + 1
        iid = idx == i
        iiid = d2d.deg[iid] == d2d.deg[iid].min()
        obs_uid[tmpj] = iid.nonzero()[0][iiid.nonzero()[0]][0]

    uidlim = d2d[obs_uid].arcsecond < 1
    if uidlim.sum() > 0:
        obs_uidlim = obs_uid[uidlim]
        ref_uidlim = ref_uid[uidlim]
        obsm1 = obsc[obs_uidlim].ra.deg
        refm1 = refc[ref_uidlim].ra.deg
        obsmd1 = obsc[obs_uidlim].dec.deg
        refmd1 = refc[ref_uidlim].dec.deg
        obsm = obsc[obs_uidlim].ra.arcsec * np.cos(
            obsc[obs_uidlim].dec.deg * np.pi / 180
        )
        refm = refc[ref_uidlim].ra.arcsec * np.cos(
            obsc[obs_uidlim].dec.deg * np.pi / 180
        )

        deltara = obsm - refm

        obsmd = obsc[obs_uidlim].dec.arcsec
        refmd = refc[ref_uidlim].dec.arcsec
        deltadec = obsmd - refmd

        clip = sigma_clip((refm - obsm), sigma=3)
        coeff0 = np.median(clip.data[~clip.mask])
        cstd = np.std(clip.data[~clip.mask])
        cobsm = obsm[~clip.mask]
        crefm = refm[~clip.mask]
        clip1 = sigma_clip(refmd - obsmd, sigma=3)
        coeff01 = np.median(clip1.data[~clip1.mask])
        cstd1 = np.std(clip1.data[~clip1.mask])
        cobsmd = obsmd[~clip1.mask]
        crefmd = refmd[~clip1.mask]
        match_num = len(obs_uidlim)

    return coeff0, coeff01, cstd, cstd1, match_num


def update_standard_output_head(
    ast_head_path,
    meanra,
    meande,
    stdra,
    stdde,
    match_num,
):
    """
    Write the standard fits header.

    Write the standard fits header from the result of Scamp and astrometry check.

    Parameters
    ----------
    ast_head_path : str
        Path of ast fits file.
    meanra : float
        Mean offset of R.A.
    meande : float
        Mean offset of Declination.
    stdra : float
        Std of offset of R.A.
    stdde : float
        Std of offset of Declination.
    match_num : int
        Number of matched stars.

    Returns
    -------
    None

    Examples
    --------
    >>> update_standard_output_head(fp_head_ast, meanra, meande, stdra, stdde, match_num)
    """
    fp_ast_head = ast_head_path
    if match_num > 0:
        header = fits.getheader(fp_ast_head, ignore_missing_simple=True)

        newheader = fits.Header()
        #        del newheader[3:22]
        newheader["STA_AST"] = (0, "Completion degree of MCI astrometric solution")
        newheader["VER_AST"] = ("v2023.01", "Version of MCI astrometry soft")
        newheader["STM_AST"] = ("2024-08-17T06:05:12.0", "Time stamp of MCI astrometry")

        newheader["EQUINOX"] = (2000.0, "mean equinox")
        newheader["RADESYS"] = ("ICRS    ", "astrometric system")
        newheader['CUNIT1'] = ('deg', 'axis unit')
        newheader['CUNIT2'] = ('deg', 'axis unit')

        newheader["CTYPE1"] = ("RA---TPV", "WCS projection type for this axis")
        newheader["CTYPE2"] = ("DEC--TPV", "WCS projection type for this axis")
        newheader["CUNIT1"] = ("deg     ", "Axis unit")
        newheader["CUNIT2"] = ("deg     ", "Axis unit")
        newheader["CRVAL1"] = (header["CRVAL1"], "World coordinate on this axis")
        newheader["CRVAL2"] = (header["CRVAL2"], "World coordinate on this axis")
        newheader["CRPIX1"] = (header["CRPIX1"], "Reference pixel on this axis")
        newheader["CRPIX2"] = (header["CRPIX2"], "Reference pixel on this axis")

        newheader["CD1_1"] = (header["CD1_1"], "Linear projection matrix")
        newheader["CD1_2"] = (header["CD1_2"], "Linear projection matrix")
        newheader["CD2_1"] = (header["CD2_1"], "Linear projection matrix")
        newheader["CD2_2"] = (header["CD2_2"], "Linear projection matrix")
        newheader["PV1_0"] = (header["PV1_0"], "Projection distortion parameter 1_0")
        newheader["PV1_1"] = (header["PV1_1"], "Projection distortion parameter 1_1")
        newheader["PV1_2"] = (header["PV1_2"], "Projection distortion parameter 1_2")
        newheader["PV1_4"] = (header["PV1_4"], "Projection distortion parameter 1_4")
        newheader["PV1_5"] = (header["PV1_5"], "Projection distortion parameter 1_5")
        newheader["PV1_6"] = (header["PV1_6"], "Projection distortion parameter 1_6")
        newheader["PV1_7"] = (header["PV1_7"], "Projection distortion parameter 1_7")
        newheader["PV1_8"] = (header["PV1_8"], "Projection distortion parameter 1_8")
        newheader["PV1_9"] = (header["PV1_9"], "Projection distortion parameter 1_9")
        newheader["PV1_10"] = (header["PV1_10"], "Projection distortion parameter 1_10")
        newheader["PV1_12"] = (header["PV1_12"], "Projection distortion parameter 1_12")
        newheader["PV1_13"] = (header["PV1_13"], "Projection distortion parameter 1_13")
        newheader["PV1_14"] = (header["PV1_14"], "Projection distortion parameter 1_14")
        newheader["PV1_15"] = (header["PV1_15"], "Projection distortion parameter 1_15")
        newheader["PV1_16"] = (header["PV1_16"], "Projection distortion parameter 1_16")
        newheader["PV2_0"] = (header["PV2_0"], "Projection distortion parameter 2_0 ")
        newheader["PV2_1"] = (header["PV2_1"], "Projection distortion parameter 2_1")
        newheader["PV2_2"] = (header["PV2_2"], "Projection distortion parameter 2_2")
        newheader["PV2_4"] = (header["PV2_4"], "Projection distortion parameter 2_4")
        newheader["PV2_5"] = (header["PV2_5"], "Projection distortion parameter 2_5")
        newheader["PV2_6"] = (header["PV2_6"], "Projection distortion parameter 2_6")
        newheader["PV2_7"] = (header["PV2_7"], "Projection distortion parameter 2_7")
        newheader["PV2_8"] = (header["PV2_8"], "Projection distortion parameter 2_8")
        newheader["PV2_9"] = (header["PV2_9"], "Projection distortion parameter 2_9")
        newheader["PV2_10"] = (header["PV2_10"], "Projection distortion parameter 2_10")
        newheader["PV2_12"] = (header["PV2_12"], "Projection distortion parameter 2_12")
        newheader["PV2_13"] = (header["PV2_13"], "Projection distortion parameter 2_13")
        newheader["PV2_14"] = (header["PV2_14"], "Projection distortion parameter 2_14")
        newheader["PV2_15"] = (header["PV2_15"], "Projection distortion parameter 2_15")
        newheader["PV2_16"] = (header["PV2_16"], "Projection distortion parameter 2_16")

        newheader["ASTIRMS1"] = (stdra, "astrom. Dispersion RMS (intern., high S/N)")
        newheader["ASTIRMS2"] = (stdde, "astrom. Dispersion RMS (intern., high S/N)")
        newheader["ASTRRMS1"] = (stdra, "astrom. Dispersion RMS (ref., high S/N)")
        newheader["ASTRRMS2"] = (stdde, "astrom. Dispersion RMS (ref., high S/N)")

        #        newheader['STA_CCRS'] = (
        #            1, "Completion degree of relative astrometric solution in MCI")
        #        newheader['VER_CCRS'] = (
        #            "v2023.01", "Version of CSST relative Astrometry soft in MCI")
        #        newheader['STM_CCRS'] = ("", "Time of last CSST Astrometry in MCI")

        newheader["ASTSIM"] = ("normal", "image classification for MCI Astr     ometry")
        newheader["ASTREF"] = ("Gaia dr3 v01", "reference catalogue for MCI Astrometry")
        newheader["ASTHIS"] = (1, "Astrometric solution Record for MCI Astrometry")
        newheader["DELT_RA"] = (meanra, "change in central RA")
        newheader["DELT_dec"] = (meande, "change in central DEC")
        newheader["DELT_ps"] = (0, "change in pixelscale")
        empty_primary = fits.PrimaryHDU(header=newheader)
        empty_primary.writeto(fp_ast_head, overwrite=True)



def copy_and_modify_header(fp_img: str, new_header: dict):
    """
    Copy the file at fp_img to a new location, modify its header,
    and return the full path of the file at the new location.

    Parameters:
    - fp_img (str): Path to the image file.
    - new_header (dict): New header information to be updated in the copied file.

    Returns:
    - new_file_path (str): Full path of the copied file at the new location.
    """
    # Get the directory containing the file
    directory = os.path.dirname(fp_img)
    
    # Construct the new destination path with parameter replaced
    new_destination_path = directory.replace("instrument", "astrometry")
    
    # Copy the file to the new destination path
    shutil.copy(fp_img, new_destination_path)

    # Construct the new file path
    file_name = os.path.basename(fp_img)
    new_file_path = os.path.join(new_destination_path, file_name)

    # Update the header of the copied file
    with fits.open(new_file_path, mode='update') as hdul:
        for key, value in new_header.items():
            hdul[1].header[key] = value
        hdul.flush()

    return new_file_path




def run_one_frame(
    refcat_now: Table,
    refcat_path: str,
    img_path: str,
    img_flag_path: str,
    img_weight_path: str,
    ast_head_path: str,
    sext_config_path: str,
    sext_out_path: str,
    sext_out_path_up: str,
    sext_para_path: str,
    sext_filter_path: str,
    scamp_config_path: str,
    scamp_head_path: str,
    scamp_head_suffix_path: str,
    scamp_head_bk_path: str,
    logger: get_logger = None,
):
    """
    Fit wcs for one chip.

    This function calculates and save the pv parameters of single chip CSST data.

    Parameters
    ----------

    refcat_now : Table
        The reference catalog.
    refcat_path : str
        The reference catalog path, *lac.fits.
    img_path : str
        The image that need to be processed, *.fits.
    img_flag_path : str
        The flag image for the image, *.fits.
    img_weight_path : str
        The weight image for the image, *.fits.
    ast_head_path : str
        The fits header combine image header and the result of astrometric processing.
    sext_config_path : str
        The config file of Source-Extractor, *.sex.
    sext_out_path : str
        The result of Source-Extractor, the type is LDAC fits, however extension is .acat.
    sext_out_path_up : str
        The updated result of Source-Extractor, the type is LDAC fits, however extension is .acat.
    sext_para_path : str
        The parameters of the output catalog, *.param.
    sext_filter_path : str
        The filter of the image, *.conv.
    scamp_config_path : str
        The config file of Scamp, *.scamp.
    scamp_head_path : str
        The Scamp head result.
    scamp_head_suffix_path : str
        The suffix Scamp head result.
    scamp_head_bk_path : str
        The standardized Scamp head result.
    logger : logging.Logger
        This is the log.

    Returns
    -------
    CsstStatus
        The status of run.

    Examples
    --------
    >>> run_one_frame(refcat_now, refcat_path, img_path, img_flag_path, img_weight_path, ast_head_path, sext_config_path, sext_out_path, sext_out_path_up, sext_para_path, sext_filter_path, scamp_config_path, scamp_head_path, scamp_head_suffix_path, scamp_head_bk_path, logger)
    """

    # set default logger
    if logger is None:
        logger = get_logger()

    logger.info("Begin of one frame.")

    # write the WCS head from *img.fits.
    fp_img = img_path
    headerkey = {
         'CD2_1': 0.0,
         'CD2_2': -1.38888888888888E-05
      }   

     
    fp_img =   copy_and_modify_header( fp_img,headerkey )
    fp_img_flag = img_flag_path
    fp_img_weight = img_weight_path

    fp_head_ast = ast_head_path


    with fits.open(fp_img) as hdul:
        print (  hdul[1].header['CD2_2']  )
    
#    initial_standard_output_head(fp_img, fp_head_ast)

    # run Source-Extractor for the image fits.
    fp_config_sext = sext_config_path
    fp_out_sext = sext_out_path
    fp_para_sext = sext_para_path
    fp_filter_sext = sext_filter_path

    

    work_sext(fp_img, fp_img_flag, fp_img_weight, fp_config_sext, fp_out_sext, fp_para_sext, fp_filter_sext)

    # update the fits of Source-Extractor
    fp_out_sext_up = sext_out_path_up
    stardata = update_sext_catalog(fp_out_sext, fp_out_sext_up)

    # run Scamp for the result from Source-Extractor.
    fp_refcat = refcat_path
    fp_config_scamp = scamp_config_path
    fp_head_scamp = scamp_head_path
    fp_head_suffix_scamp = scamp_head_suffix_path
    work_scamp(
        fp_out_sext_up, fp_refcat, fp_config_scamp, fp_head_scamp, fp_head_suffix_scamp
    )

    # rewrite the Scamp WCS head
    fp_head_bk_scamp = scamp_head_bk_path
    rewrite_scamp_head_to_standard(fp_head_scamp, fp_head_bk_scamp)

    #
    wcshdr = fits.getheader(fp_head_bk_scamp, ignore_missing_simple=True)
    if not wcshdr.get("PV1_16"):
        print("Warning: astrometry failure in this chip. Exiting...")
        os.system(" rm " + fp_head_bk_scamp)
    else:
        os.system("cp " + fp_head_bk_scamp + " " + fp_head_ast)

    wcshdr["CTYPE1"] = "RA---TPV"
    wcshdr["CTYPE2"] = "DEC--TPV"
    w = WCS(wcshdr)
    meanra, meande, stdra, stdde, match_num = check_astrometry_result(
        w, stardata, refcat_now
    )

    logger.info("################# astrometry first result: ##############")
    logger.info(" The first step: check match ")
    if match_num < 6:
        logger.info(" match_num less than 10 bad astrometry")
        return CsstStatus.ERROR
    else:
        logger.info(" check error")
        print(meanra, meande, stdra, stdde, match_num)
        if (
            meanra * 1000 < 1000
            and meande * 1000 < 1000
            and stdra * 1000.0 < 3000
            and stdde * 1000 < 3000
        ):
            logger.info(" good result for " + str(fp_img))
            logger.info(
                "median ra_off, dec_off (mas) from Scamp:"
                + str(meanra * 1000.0)
                + "  "
                + str(meande * 1000.0)
            )
            logger.info(
                "rms ra_off,   dec_off (mas) from Scamp:"
                + str(stdra * 1000.0)
                + "  "
                + str(stdde * 1000.0)
            )

        else:
            logger.error("step3 big error")
            return CsstStatus.ERROR

    update_standard_output_head(fp_head_ast, meanra, meande, stdra, stdde, match_num)
    logger.info("The fitting ran smoothly.")
    return CsstStatus.PERFECT


def run_mci_astrometry(dm: CsstMCIDataManager, logger: get_logger = None):
    """
    Fit wcs for more than one chips.

    This function calculates and save the pv parameters of many chips of CSST data.

    Parameters
    ----------
    dm : CsstMCIDataManager
        The object from CsstMCIDataManager class.
    logger : logging.Logger
        This is the log.

    Returns
    -------
    CsstStatus
        The status of run.

    Examples
    --------
    >>> result = run_mci_astrometry(logger=None)
    """
    from . import PACKAGE_PATH

    CONFIG_PATH = PACKAGE_PATH + "/data/"

    # set default logger
    if logger is None:
        logger = get_logger()

    # get information of input image
    logger.info("Begin of csst_mci_astrometry.")

    # get the image fits
    dm.target_detectors = ['C1']
    img_path = dm.l1_detector(detector=dm.target_detectors[0], post="img.fits")
    image_header = fits.getheader(img_path, ext=0)

    # the point towards the sky from the image fits
    pointing_ra = image_header["RA_OBJ"]
    pointing_dec = image_header["DEC_OBJ"]
    refcat = gaiadr3_query(pointing_ra, pointing_dec, rad=2.0)
    # saved catalog as gaiadr3.fits
    fp_refcat = dm.l1_file(name="gaiadr3.fits")
    # This step only do once for one exposure
    refcat.write(fp_refcat, format="fits", overwrite=True)
    # change epoch of the catalog
    refcat_now = update_catalog_space_motion(refcat, image_header, dfsapi=False)
    refcat_now = refcat_now[~np.isnan(refcat_now["X_WORLD"])]
    fp_refcat_now = dm.l1_file(name="gaiadr3now.fits")
    refcat_now.write(fp_refcat_now, format="fits", overwrite=True)

    hdu = fits.open(fp_refcat_now)
    hdu1 = convert_hdu_to_ldac(hdu)
    hdup = fits.PrimaryHDU()
    hdu = hdu1[0]
    tbhdu = hdu1[1]
    thdulist = fits.HDUList([hdup, hdu, tbhdu])
    fp_refcat_lac_now = dm.l1_file(name="gaiadr3nowlac.fits")
    thdulist.writeto(fp_refcat_lac_now, overwrite=True)

    # do distortion resolving for each detector in parallel
    results = []
    for detector in dm.target_detectors:
        result = run_one_frame(
            refcat_now,
            refcat_path=dm.l1_file(name="gaiadr3nowlac.fits"),
            img_path=dm.l1_detector(detector=detector, post="img.fits"),
            img_flag_path=dm.l1_detector(detector=detector, post="flg.fits"),
            img_weight_path=dm.l1_detector(detector=detector, post="wht.fits"),
            ast_head_path=dm.l1_detector(detector=detector, post="ast.fits").replace("instrument", "astrometry"),
            sext_config_path=CONFIG_PATH + "astrom.sex",
            sext_out_path=dm.l1_detector(detector=detector, post="nimg.acat").replace("instrument", "astrometry"),
            sext_out_path_up=dm.l1_detector(detector=detector, post="ste.acat").replace("instrument", "astrometry"),
            sext_para_path=CONFIG_PATH + "astrom.param",
            sext_filter_path=CONFIG_PATH + "gauss_4.0_7x7.conv",
            scamp_config_path=CONFIG_PATH + "g3default.scamp",
            scamp_head_path=dm.l1_detector(detector=detector, post="ste.head").replace("instrument", "astrometry"),
            scamp_head_suffix_path=".head",
            scamp_head_bk_path=dm.l1_detector(detector=detector, post="bk.head").replace("instrument", "astrometry"),
            logger=logger,
        )
        results.append(result)

    return (
        CsstStatus.PERFECT
        if all([_ == CsstStatus.PERFECT for _ in results])
        else CsstStatus.ERROR
    )

