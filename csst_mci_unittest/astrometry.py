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
"""
import os
import sys
import time
import math
import random
import numpy as np
import scipy.spatial
from scipy.spatial.distance import cdist

from astropy import units as u
from astropy.table import Table, Column, MaskedColumn
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, Distance, GCRS, CartesianRepresentation
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy.stats import sigma_clip
from astroquery.vizier import Vizier

from csst_mci_common.data_manager import CsstMCIDataManager
from csst_mci_common.logger import get_logger
from csst_mci_common.status import CsstStatus

import logging
import warnings
from multiprocessing import Process, Manager, Pool
import joblib


def up_imgfits_head(imgfits, nimgfits):
    hdu = fits.open(imgfits)
    header2 = hdu[1].header
    header2["CD2_2"] = -1 * header2["CD2_2"]
    hdu.writeto(nimgfits, overwrite="True")


def gaiadr3_query(
    ra: list,
    dec: list,
    rad: float = 1.0,
    maxmag: float = 25,
    maxsources: float = 1000000,
):
    """
    Acquire the Gaia DR3, from work of zhang tianmeng.

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


def convert_hdu_to_ldac(hdu):
    """
    Convert hdu table to ldac format.

    Convert an hdu table to a fits_ldac table (format used by astromatic suite).

    Parameters
    ----------
    hdu : astropy.io.fits.hdu.hdulist.HDUList
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
    col1 = fits.Column(name="Field Header Card", array=tblhdr, format="13200A")
    cols = fits.ColDefs([col1])
    tbl1 = fits.BinTableHDU.from_columns(cols)
    tbl1.header["TDIM1"] = "(80,   {0})".format(len(hdu[1].header))
    tbl1.header["EXTNAME"] = "LDAC_IMHEAD"

    dcol = fits.ColDefs(hdu[1].data)
    tbl2 = fits.BinTableHDU.from_columns(dcol)
    tbl2.header["EXTNAME"] = "LDAC_OBJECTS"
    return (tbl1, tbl2)


def saveresult(fp_ast_head, meanra, meande, stdra, stdde, match_num):
    """
    Write the standard fits header.

    Write the standard fits header from the result of scamp and astrometry check.

    Parameters
    ----------
    fp_ast_head : str
        Path of ast fits file.
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

    Returns
    -------
    None

    Examples
    --------
    >>> saveresult(fp_ast_head, meanra, meande, stdra, stdde, match_num)
    """
    # wcshead = head + '.fits'
    if match_num > 0:
        header = fits.getheader(fp_ast_head, ignore_missing_simple=True)

        newheader = fits.Header()
        #        del newheader[3:22]
        newheader["STA_AST"] = (0, "Completion degree of MCI astrometric solution")
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
        newheader["PV1_0"] = (header["PV1_0"], "Projection distortion parameter")
        newheader["PV1_1"] = (header["PV1_1"], "Projection distortion parameter")
        newheader["PV1_2"] = (header["PV1_2"], "Projection distortion parameter")
        newheader["PV1_4"] = (header["PV1_4"], "Projection distortion parameter")
        newheader["PV1_5"] = (header["PV1_5"], "Projection distortion parameter")
        newheader["PV1_6"] = (header["PV1_6"], "Projection distortion parameter")
        newheader["PV1_7"] = (header["PV1_7"], "Projection distortion parameter")
        newheader["PV1_8"] = (header["PV1_8"], "Projection distortion parameter")
        newheader["PV1_9"] = (header["PV1_9"], "Projection distortion parameter")
        newheader["PV1_10"] = (header["PV1_10"], "Projection distortion parameter")
        newheader["PV1_12"] = (header["PV1_12"], "Projection distortion parameter")
        newheader["PV1_13"] = (header["PV1_13"], "Projection distortion parameter")
        newheader["PV1_14"] = (header["PV1_14"], "Projection distortion parameter")
        newheader["PV1_15"] = (header["PV1_15"], "Projection distortion parameter")
        newheader["PV1_16"] = (header["PV1_16"], "Projection distortion parameter")
        newheader["PV2_0"] = (header["PV2_0"], "Projection distortion parameter")
        newheader["PV2_1"] = (header["PV2_1"], "Projection distortion parameter")
        newheader["PV2_2"] = (header["PV2_2"], "Projection distortion parameter")
        newheader["PV2_4"] = (header["PV2_4"], "Projection distortion parameter")
        newheader["PV2_5"] = (header["PV2_5"], "Projection distortion parameter")
        newheader["PV2_6"] = (header["PV2_6"], "Projection distortion parameter")
        newheader["PV2_7"] = (header["PV2_7"], "Projection distortion parameter")
        newheader["PV2_8"] = (header["PV2_8"], "Projection distortion parameter")
        newheader["PV2_9"] = (header["PV2_9"], "Projection distortion parameter")
        newheader["PV2_10"] = (header["PV2_10"], "Projection distortion parameter")
        newheader["PV2_12"] = (header["PV2_12"], "Projection distortion parameter")
        newheader["PV2_13"] = (header["PV2_13"], "Projection distortion parameter")
        newheader["PV2_14"] = (header["PV2_14"], "Projection distortion parameter")
        newheader["PV2_15"] = (header["PV2_15"], "Projection distortion parameter")
        newheader["PV2_16"] = (header["PV2_16"], "Projection distortion parameter")

        newheader["ASTIRMS1"] = (stdra, "Astrom. dispersion RMS (intern., high S/N)")
        newheader["ASTIRMS2"] = (stdde, "Astrom. dispersion RMS (intern., high S/N)")
        newheader["ASTRRMS1"] = (stdra, "Astrom. dispersion RMS (ref., high S/N)")
        newheader["ASTRRMS2"] = (stdde, "Astrom. dispersion RMS (ref., high S/N)")

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
        newheader["DELT_RA"] = (meanra, "Change in central RA")
        newheader["DELT_dec"] = (meande, "Change in central DEC")
        newheader["DELT_ps"] = (0, "Change in pixelscale")
        empty_primary = fits.PrimaryHDU(header=newheader)
        empty_primary.writeto(fp_ast_head, overwrite=True)


def write_wcs_head(head, output):
    """
    Write the WCS head.

    Write the WCS head from Scamp to the standard fits header.

    Parameters
    ----------
    head : str
        The name of the fits need to calculate the distortion, *img.fits.
    output : str
        A new head file from head.
    Returns
    -------
    None

    Examples
    --------
    >>> write_wcs_head(head, output)
    """
    # wcshead = head + '.fits'
    if head[-4:] == "fits":
        header = fits.getheader(head, 1)
        w = WCS(header)
        print(
            w,
            "                                                                                aiyouwei W",
        )
        #        hdu = w.to_fits()
        #        newheader= hdu[0].header
        newheader = fits.Header()
        print(newheader)
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
        empty_primary.writeto(output, overwrite=True)


def refframe_tran(catalog, header, dfsapi: bool = True):
    if dfsapi:
        ra = catalog["Ra"]  # For DFS api
        dec = catalog["Dec"]
        parallax = catalog["Parallax"]
        pmra = catalog["Pmra"]  # .value
        pmdec = catalog["Pmdec"]  # .value
    else:
        ra = catalog["RA_ICRS"]  # .value
        dec = catalog["DE_ICRS"]  # .value
        parallax = catalog["Plx"]  # .value
        pmra = catalog["pmRA"]  # .value
        pmdec = catalog["pmDE"]  # .value

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
    c = SkyCoord(0 * u.deg, 0 * u.deg)
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

    if dfsapi:
        catalog["Ra"][cut] = c_epoch_now.ra.degree  # For DFS api
        catalog["Dec"][cut] = c_epoch_now.dec.degree
        catalog["Parallax"][cut] = (
            c_epoch_now.distance.to(u.arcsec, equivalencies=u.parallax()) * 1000
        )
        catalog["Pmra"][cut] = c_epoch_now.pm_ra_cosdec
        catalog["Pmdec"][cut] = c_epoch_now.pm_dec
        catalog.rename_column("Ra", "X_WORLD")
        catalog.rename_column("Dec", "Y_WORLD")
        col_errra = Column(name="ERRA_WORLD", data=catalog["Gmag"] / 1000 / 3600)
        col_errde = Column(name="ERRB_WORLD", data=catalog["Gmag"] / 1000 / 3600)
        catalog.rename_column("Gmag", "MAG")
        catalog.add_column(col_errra)
        catalog.add_column(col_errde)

    else:
        catalog["RA_ICRS"][cut] = c_epoch_now.ra.degree
        catalog["DE_ICRS"][cut] = c_epoch_now.dec.degree
        catalog["Plx"][cut] = (
            c_epoch_now.distance.to(u.arcsec, equivalencies=u.parallax()) * 1000
        )
        catalog["pmRA"][cut] = c_epoch_now.pm_ra_cosdec
        catalog["pmDE"][cut] = c_epoch_now.pm_dec
        catalog.rename_column("RA_ICRS", "X_WORLD")
        catalog.rename_column("DE_ICRS", "Y_WORLD")
        col_errra = Column(name="ERRA_WORLD", data=catalog["Gmag"] / 1000 / 3600)
        col_errde = Column(name="ERRB_WORLD", data=catalog["Gmag"] / 1000 / 3600)
        catalog.rename_column("Gmag", "MAG")
        catalog.add_column(col_errra)
        catalog.add_column(col_errde)

    return catalog


def rewrite_wcs_head(head, output):
    """
    Rewrite the WCS head.

    Rewrite the WCS head from Scamp to the standard fits header.

    Parameters
    ----------
    head : str
        Scamp head file name.
    output : str
        Rewrited scamp head file name.

    Returns
    -------
    None

    Examples
    --------
    >>> rewrite_wcs_head(head, output)
    """
    # wcshead = head + '.fits'

    if head[-4:] == "head":
        f = open(head, "r")
        f1 = open(output, "w")
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
        f1.write(a.replace("\n", ""))
        f1.close()
        f.close()
    return


def createxy_upwcs(fileout: str, filestar: str, fitsname: str):
    print(fileout)
    hdulist = fits.open(fileout, mode="update")
    ptdata1 = hdulist[2].data
    ptdata1["FLUXERR_AUTO"] = ptdata1["FLUX_AUTO"] / 100
    tdata = ptdata1
    region = (
        (tdata["AWIN_IMAGE"] < 0.8)
        & (tdata["AWIN_IMAGE"] > 0.6)
        & (tdata["BWIN_IMAGE"] < 0.8)
        & (tdata["BWIN_IMAGE"] > 0.6)
    )

    region = (tdata["AWIN_IMAGE"] < 1.2) & (tdata["AWIN_IMAGE"] > 0.2)
    tdata = tdata[region]
    if len(tdata["ELLIPTICITY"]) > 800:
        a = np.sort(tdata, order=["FLUX_AUTO"], axis=0)
        tdata = a[-150:-1]
    new = tdata["FLUX_AUTO"]

    filesatreg = fileout + ".reg"
    cmd = "rm " + filesatreg
    os.system(cmd)
    f3 = open(filesatreg, "a")
    f3.write(
        'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include =1 source=1"
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

    hdulist[2].data = tdata
    hdulist.flush()
    hdulist.writeto(filestar, overwrite="True")
    return tdata


def solvefield(fileseout, gaialacnm, refpar, fp_scamp_head, fp_up_head):
    """
    Do astrometry by scamp.

    Do astrometry by scamp from the result of source-extractor.

    Parameters
    ----------
    fileseout : str
        Fits name of result of source-extractor
    gaialacnm : str
        Gaia reference catalog
    refpar : str
        Scamp config file
    fp_scamp_head : str
        Scamp result head file
    fp_up_head : str
        The aheader suffix file

    Returns
    -------
    None

    Examples
    -------
    >>> solvefield(fileseout, gaialacnm, refpar, fp_scamp_head, fp_up_head)
    """

    cmd_solve = (
        " scamp "
        + fileseout
        + "  -c  "
        + refpar
        + "  -ASTREFCAT_NAME  "
        + gaialacnm
        + " -AHEADER_SUFFIX "
        + fp_up_head
    )
    os.system(cmd_solve)
    print(cmd_solve)
    try:
        assert os.access(fp_scamp_head, os.F_OK)
    except AssertionError:
        raise AssertionError("no scamp result head file")


def check_astrometry(w, obsdata, refdata):
    coeff0 = coeff01 = 99999
    cstd = cstd1 = 99999
    match_num = 0
    obsc = w.pixel_to_world(obsdata["XWIN_IMAGE"] - 1, obsdata["YWIN_IMAGE"] - 1)
    mask = abs(refdata["X_WORLD"]) > 0  # and (abs( refdata['Y_WORLD'] >0 ))
    refdata = refdata[mask]
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


def work_sext(fitsname: str, fwhtname: str, fileout: str, filestar: str):
    from . import PACKAGE_PATH

    CONFIG_PATH = PACKAGE_PATH + "/data/"

    cmd_sex = (
        "source-extractor  -c "
        + CONFIG_PATH
        + "astrom.sex "
        + fitsname
        + " -WEIGHT_IMAGE "
        + fwhtname
        + " -CATALOG_NAME "
        + fileout
        + " -PARAMETERS_NAME "
        + CONFIG_PATH
        + "astrom.param"
        + " -FILTER_NAME "
        + CONFIG_PATH
        + "gauss_4.0_7x7.conv"
    )
    print(cmd_sex)
    os.system(cmd_sex)
    #    stardata = createxy(fileout, filestar)
    stardata = createxy_upwcs(fileout, filestar, fitsname)
    return stardata


def work_scamp(
    fitsname: str,
    filestar: str,
    gaialacnm: str,
    refpar: str,
    fp_scamp_head: str,
    fp_up_head: str,
    fp_ast_head: str,
    refcat_now,
    stardata,
    fp_bk_head,
):
    solvefield(filestar, gaialacnm, refpar, fp_scamp_head, fp_up_head)
    rewrite_wcs_head(fp_scamp_head, fp_bk_head)
    wcshdr = fits.getheader(fp_bk_head, ignore_missing_simple=True)

    if not wcshdr.get("PV1_16"):
        print("Warning: astrometry failure in this chip. Exiting...")
        cmd = " rm " + fp_bk_head
        os.system(cmd)
        # sys.exit(1)
        # exit(0)
    else:
        cmd = "cp " + fp_bk_head + " " + fp_ast_head
        os.system(cmd)

    #    try:
    #        assert (wcshdr.get("PV1_16"))
    #    except AssertionError as e:
    #        print(f"Warning: {e}")
    wcshdr["CTYPE1"] = "RA---TPV"
    wcshdr["CTYPE2"] = "DEC--TPV"
    w = WCS(wcshdr)
    meanra, meande, stdra, stdde, match_num = check_astrometry(w, stardata, refcat_now)
    return meanra, meande, stdra, stdde, match_num


def singlechip_wcsfit(
    dm: CsstMCIDataManager,
    detector: int,
    refcat_now: Table,
    fitsname: str,
    fwhtname: str,
    refname: str,
):
    """
    Save the parameters for writing the standard fits header.

    Get the parameters from the result of scamp and astrometry check, and write the standard fits header.

    Parameters
    ----------
    dm : CsstMCIDataManager
        Data manager of this pipeline.
    detector : int
        Detector index.
    refcat_now : Table
        The reference catalog at specified epoch by apply the space motion.
    fitsname : str
        The name of the fits need to calculate the distortion, *img.fits， the fits file need to have
        initial wcs like CRVAL, CRPIX and CD{?}_{?}.
    fwhtname : str
        The name of the fits contained the weight information.
    refname : str
        The name of the reference catalog.

    Returns
    -------
    CsstStatus
        The status of run.

    Examples
    -------
    >>> result = singlechip_wcsfit(dm, detector, refcat_now, fitsname, fwhtname, refname)
    """
    from . import PACKAGE_PATH

    CONFIG_PATH = PACKAGE_PATH + "/data/"
    logger = get_logger()
    fp_filestar = dm.l1_detector(
        detector=detector, post="ste.acat"
    )  # dm.l1_file(name="star.acat")
    fp_fileout = dm.l1_detector(
        detector=detector, post="nimg.acat"
    )  # dm.l1_file(name=".acat")
    fp_scamp_head = dm.l1_detector(
        detector=detector, post="ste.head"
    )  # dm.l1_file(name=".head")
    fp_ast_head = dm.l1_detector(
        detector=detector, post="ast.fits"
    )  # dm.l1_file(name="ast.fits")
    fp_bk_head = dm.l1_detector(
        detector=detector, post="bk.head"
    )  # dm.l1_file(name="ast.fits")

    if (
        detector == 12
        or detector == 13
        or detector == 18
        or detector == 19
        or detector == 25
    ):
        defpars2file = "g5default.scamp"
        defpars3file = "g6default.scamp"
    else:
        defpars2file = "nmg5default.scamp"
        defpars3file = "nmg6default.scamp"

    write_wcs_head(fitsname, fp_ast_head)
    refpar = CONFIG_PATH + "g3default.scamp"
    print(fp_filestar, fp_fileout, fp_scamp_head, fp_ast_head)
    fp_up_head = ".ahead"

    stardata = work_sext(fitsname, fwhtname, fp_fileout, fp_filestar)
    meanra, meande, stdra, stdde, match_num = work_scamp(
        fitsname,
        fp_filestar,
        refname,
        refpar,
        fp_scamp_head,
        fp_up_head,
        fp_ast_head,
        refcat_now,
        stardata,
        fp_bk_head,
    )

    logger.info("################# astrometry first result: ##############")
    logger.info(" The first step: check match ")
    if match_num < 6:
        logger.info(" match_num  less than 10   bad astrometry")
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
            logger.info(" good result for " + str(fitsname))
            logger.info(
                "median ra_off,   dec_off (mas) from scamp:"
                + str(meanra * 1000.0)
                + "  "
                + str(meande * 1000.0)
            )
            logger.info(
                "rms ra_off,   dec_off (mas) from scamp:"
                + str(stdra * 1000.0)
                + "  "
                + str(stdde * 1000.0)
            )
            """
            refpar = CONFIG_PATH + defpars2file
            fp_up_head = dm.l1_detector(detector=detector, post="ste.head")
            meanra, meande, stdra, stdde, match_num = work_scamp(
                fitsname, fwhtname, fp_fileout, fp_filestar, refname,
                refpar, fp_scamp_head, fp_up_head, fp_ast_head, refcat_now, stardata,fp_bk_head)
            logger.info(
                "################# astrometry second  result: ##############")
            logger.info(" The first step: check match ")
            if (match_num < 6):
                logger.info(" match_num  less than 10   bad astrometry")
                return CsstStatus.ERROR

            else:
                logger.info("The second step: check error")
                if (meanra * 1000 < 30 and meande * 1000 <
                        30 and stdra * 1000. < 600 and stdde * 1000 < 600):
                    logger.info(" good result for " + str(fitsname))
                    logger.info('median ra_off,   dec_off (mas) from scamp:' +
                                str(meanra * 1000.) + "  " + str(meande * 1000.))
                    logger.info('rms ra_off,   dec_off (mas) from scamp:' +
                                str(stdra * 1000.) + "  " + str(stdde * 1000.))

                    refpar = CONFIG_PATH + defpars3file
                    fp_up_head = dm.l1_detector(
                        detector=detector, post="ste.head")
                    meanra, meande, stdra, stdde, match_num = work_scamp(
                        fitsname, fwhtname, fp_fileout, fp_filestar, refname, refpar,
                        fp_scamp_head, fp_up_head, fp_ast_head, refcat_now, stardata,fp_bk_head)
                    logger.info(
                        "################# astrometry third result: ##############")
                    logger.info(" The first step: check match ")
                    if (match_num < 6):
                        logger.info(
                            " match_num  less than 10   bad astrometry")
                        return CsstStatus.ERROR
                    else:
                        logger.info("The second step: check error")
                        if (meanra * 1000 < 30 and meande * 1000 <
                                30 and stdra * 1000. < 300 and stdde * 1000 < 300):
                            logger.info(" good result for " + str(fitsname))
                            logger.info('median ra_off,   dec_off (mas) from scamp:' +
                                        str(meanra * 1000.) + "  " + str(meande * 1000.))
                            logger.info('rms ra_off,   dec_off (mas) from scamp:' +
                                        str(stdra * 1000.) + "  " + str(stdde * 1000.))

                            return meanra, meande, stdra, stdde, match_num
                        else:
                            logger.error("step1 big  error")
                            return CsstStatus.ERROR

                else:
                    logger.error(" step2 big   error")
                    return CsstStatus.ERROR
                """
        else:
            logger.error("step3 big  error")
            return CsstStatus.ERROR

    saveresult(fp_ast_head, meanra, meande, stdra, stdde, match_num)


def run_one_frame(
    dm: CsstMCIDataManager,
    detector: int,
    refcat_now: Table,
    use_dfs: bool = False,
    logger: logging.Logger = None,
):
    """
    Fit wcs for one chip.

    This function calculates and save the pv parameters of single chip CSST data.

    Parameters
    ----------
    dm : CsstMCIDataManager
        Data manager of this pipeline.
    detector : int
        Detector number.
    refcat_now : Table
        The reference catalog.
    use_dfs : bool
        If Ture, using DFS api, if False, using gaia_query.
    logger : logging.Logger
        This is the log.

    Returns
    -------
    CsstStatus
        The status of run.

    Examples
    --------
    >>> result = run_one_frame(dm, detector, refcat_now, use_dfs= False, logger= None)
    """

    # set default logger
    if logger is None:
        logger = get_logger()

    if use_dfs:
        ref_name = "gaia_dfs.fits"
    else:
        ref_name = "gaiadr3nowlac.fits"

    this_header = fits.getheader(
        filename=dm.l1_detector(detector=detector, post="nimg.fits"), ext=1
    )

    result = singlechip_wcsfit(
        dm,
        detector,
        refcat_now,
        fitsname=dm.l1_detector(detector=detector, post="nimg.fits"),
        fwhtname=dm.l1_detector(detector=detector, post="wht.fits"),
        refname=dm.l1_file(name=ref_name),
    )
    if result == CsstStatus.ERROR:
        logger.error("Something went wrong.")
        return CsstStatus.ERROR
    else:
        logger.info("The fitting ran smoothly.")
        return CsstStatus.PERFECT  # , meanra,meande,stdra,stdde,match_num


def run_mci_astrometry(
    dm: CsstMCIDataManager,
    use_dfs: bool = False,
    logger: logging.Logger = None,
    debug: bool = False,
):
    """
    Fit wcs for more than one chips.

    This function calculates and save the pv parameters of many chips of CSST data.

    Parameters
    ----------
    dm : CsstMCIDataManager
        Data manager of this pipeline.
    use_dfs : bool
        If Ture, using DFS api, if False, using gaia_query.
    logger : logging.Logger
        This is the log.
    debug : bool
        If Ture, return CsstStatus, filerecorder and other results, if False, only return CsstStatus.

    Returns
    -------
    CsstStatus
        The status of run.

    Examples
    --------
    >>> result = run_mci_astrometry(dm, use_dfs=False, logger=None, debug=False)
    """

    # set default logger
    if logger is None:
        logger = get_logger()

    imgfits = dm.l1_detector(detector=dm.target_detectors[0], post="img.fits")
    nimgfits = dm.l1_detector(detector=dm.target_detectors[0], post="nimg.fits")

    print(imgfits, nimgfits)
    up_imgfits_head(imgfits, nimgfits)

    header = fits.getheader(dm.l1_detector(detector=dm.target_detectors[0]), ext=0)
    pointing_ra = header["OBJ_RA"]
    pointing_dec = header["OBJ_DEC"]

    if use_dfs:
        # saved catalog as gaia_dfs.fits
        fp_refcat = dm.l1_file(name="gaia_dfs.fits")

    else:
        # get gaia catalog from astroquery
        refcat = gaiadr3_query(pointing_ra, pointing_dec, rad=2.0)
        # saved catalog as gaiadr3.fits
        fp_refcat = dm.l1_file(name="gaiadr3.fits")
        # This step only do once for one exposure
        refcat.write(fp_refcat, format="fits", overwrite=True)
        # not change epoch in this cycle
        refcat_now = refframe_tran(refcat, header, dfsapi=use_dfs)
        fp_refcat_now = dm.l1_file(name="gaiadr3now.fits")
        refcat_now.write(fp_refcat_now, format="fits", overwrite=True)
        hdu = fits.open(fp_refcat_now)
        hdu1 = convert_hdu_to_ldac(hdu)

        hdup = fits.PrimaryHDU()
        hdu = hdu1[0]
        tbhdu = hdu1[1]
        thdulist = fits.HDUList([hdup, hdu, tbhdu])
        fp_refcat_lac_now = dm.l1_file(
            name="gaiadr3nowlac.fits"
        )  # saved catalog as gaiadr3.fits
        # This step only do once for one exposure
        thdulist.writeto(fp_refcat_lac_now, overwrite=True)

    print(dm.target_detectors)
    # do distortion resolving for each detector in parallel
    results = joblib.Parallel(n_jobs=len(dm.target_detectors))(
        joblib.delayed(run_one_frame)(dm, detector, refcat_now, use_dfs=use_dfs)
        for detector in dm.target_detectors
    )

    if debug:
        return (
            CsstStatus.PERFECT
            if all([_ == CsstStatus.PERFECT for _ in results])
            else CsstStatus.ERROR
        )
    else:
        return (
            CsstStatus.PERFECT
            if all([_ == CsstStatus.PERFECT for _ in results])
            else CsstStatus.ERROR
        )
