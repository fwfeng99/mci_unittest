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
    2024-05-29, Xiyan Peng, modified
    2024-06-04, Xiyan Peng, modified
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

    Uses astroquery.vizier to query the Gaia Data Release 3 (DR3) catalog based on given right ascension (RA), declination (Dec), and other criteria.

    Parameters
    ----------
    ra : list
        RA of center in degrees.
    dec : list
        Dec of center in degrees.
    rad : float
        Search radius in degrees.
    maxmag : float
        Maximum magnitude limit.
    maxsources : float
        Maximum number of sources to return.

    Returns
    -------
    Table
        table of reference catalog.

    Examples
    --------
    >>> catalog = gaiadr3_query(ra, dec, rad, maxmag, maxsources)
    """
    # Astrometry-related parameters and constraints
    vquery = Vizier(
        columns=["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Plx", "RVDR2", "Gmag"],
        row_limit=maxsources,
        column_filters={"Gmag": ("<%f" % maxmag), "Plx": ">0"},
    )

    # The GaiaDR3 sources in the sky field
    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame="icrs")

    # The GaiaDR3 sources in the sky field
    result = vquery.query_region(coord, radius=rad * u.deg, catalog="I/355/gaiadr3")
    table = result[0]

    return table


def update_catalog_space_motion(
    catalog: Table,
    header: fits.header.Header,
    dfsapi: bool = False
):
    """
    Update the spatial information of objects in the reference catalog.

    Updates the positions of objects in the reference catalog to the observed epoch, considering proper motion and parallax.
    It supports using either the GAIA DR3 catalog directly or through the DFS API.

    Parameters
    ----------
    catalog : Table
        The original reference catalog, containing object information such as position and proper motion.
    header : fits.header.Header
        The header information of the FITS file, containing the observation epoch and other relevant astrometric parameters.
    dfsapi : bool, optional
        A flag indicating whether to use the DFS API to query data. Defaults to False, indicating direct use of the GAIA DR3 catalog.

    Returns
    -------
    Table
        table of update reference catalog.

    Examples
    --------
    >>> catalog = gaiadr3_query(ra, dec, rad, maxmag, maxsources)
    """
    # if dfsapi:
    #     ra = catalog["Ra"]  # For DFS api
    #     dec = catalog["Dec"]
    #     parallax = catalog["Parallax"]
    #     pmra = catalog["Pmra"]
    #     pmdec = catalog["Pmdec"]
    #     mag = catalog["Gmag"]
    # else:

    # Extract celestial coordinates and other astrometric information from the catalog
    ra = catalog["RA_ICRS"]
    dec = catalog["DE_ICRS"]
    parallax = catalog["Plx"]
    pmra = catalog["pmRA"]
    pmdec = catalog["pmDE"]
    mag = catalog["Gmag"]

    # Create a time object based on the exposure time
    dt2 = TimeDelta(header["EXPTIME"], format="sec")
    # Create a time object based on the observation date
    t = Time(header["DATE-OBS"], format="isot", scale="utc")
    # Current epoch
    epoch_now = t  # + dt2

    # Extract position and velocity data of satellite from the header
    input_x = header["POSI0_X"]
    input_y = header["POSI0_Y"]
    input_z = header["POSI0_Z"]
    input_vx = header["VELO0_X"]
    input_vy = header["VELO0_Y"]
    input_vz = header["VELO0_Z"]

    # Create a Cartesian representation for the position of satellite
    tt1 = CartesianRepresentation(
        input_x * 1000 * u.m,
        input_y * 1000 * u.m,
        input_z * 1000 * u.m,
    )
    # Create a Cartesian representation for the velocity of satellite
    tt2 = CartesianRepresentation(
        input_vx * 1000 * u.m / u.s,
        input_vy * 1000 * u.m / u.s,
        input_vz * 1000 * u.m / u.s,
    )

    # Filter out the catalog with non-zero proper motion components
    cut = (abs(pmra) > 0) & (abs(pmdec) > 0)
    catpm = catalog[cut]

    # Create a SkyCoord object with RA, Dec, distance, proper motion, and reference time
    c = SkyCoord(
        ra=ra[cut],
        dec=dec[cut],
        distance=Distance(parallax=abs(parallax[cut]) * u.mas),
        pm_ra_cosdec=pmra[cut],
        pm_dec=pmdec[cut],
        obstime=Time(2016.0, format="jyear", scale="utc"),
        frame="icrs",
    )

    # Set the reference epoch to 2000
    epochobs = Time(2000.0, format="jyear", scale="utc")
    # Compute the celestial positions at the epoch
    c_epoch_now = c.apply_space_motion(epochobs)

    # Create a new table with updated astrometric information
    catalog_now = Table()
    # Add updated RA, Dec, proper motion, parallax, and magnitude to the new table
    catalog_now.add_column(Column(name="X_WORLD", data=c_epoch_now.ra.degree, unit=u.deg))
    catalog_now.add_column(Column(name="Y_WORLD", data=c_epoch_now.dec.degree, unit=u.deg))
    catalog_now.add_column(Column(name="pmRA", data=c_epoch_now.pm_ra_cosdec, unit=u.mas / u.yr))
    catalog_now.add_column(Column(name="pmDE", data=c_epoch_now.pm_dec, unit=u.mas / u.yr))
    catalog_now.add_column(Column(name="Plx", data=c_epoch_now.distance.to(u.arcsec, equivalencies=u.parallax()), unit=u.mas))
    catalog_now.add_column(Column(name="MAG", data=mag[cut], unit=u.mag))
    # Add magnitude error columns
    catalog_now.add_column(Column(name="ERRA_WORLD", data=mag[cut] / 1000 / 3600))
    catalog_now.add_column(Column(name="ERRB_WORLD", data=mag[cut] / 1000 / 3600))

    # Return the updated catalog
    return catalog_now


def convert_hdu_to_ldac(hdu: fits.hdu.hdulist.HDUList):
    """
    Convert hdu table to LDAC format.

    Convert the HDUList into the LDAC format. The conversion primarily involves adjusting the header information and data format to comply with the LDAC specification.

    Parameters
    ----------
    hdu : fits.hdu.hdulist.HDUList
        The HDUList object to be converted.

    Returns
    -------
    tuple:
        The tuple contains Header info for fits table (LDAC_IMHEAD) and Data table (LDAC_OBJECTS),
            the type of table is fits.BinTableHDU.

    Examples
    --------
    >>> tbl = convert_hdu_to_ldac(hdu)
    """
    # Convert the header of the second HDU to a numpy array and define it as a column in the new table
    tblhdr = np.array([hdu[1].header.tostring()])
    # Define the column properties and create a new table HDU for the header information
    hcol = fits.ColDefs([fits.Column(name="Field Header Card", array=tblhdr, format="13200A")])
    tbl1 = fits.BinTableHDU.from_columns(hcol)
    # Set the TDIM1 keyword to describe the dimension of the header column and set the EXTNAME keyword to LDAC_IMHEAD
    tbl1.header["TDIM1"] = "(80,   {0})".format(len(hdu[1].header))
    tbl1.header["EXTNAME"] = "LDAC_IMHEAD"

    # Directly use the column properties of the second HDU data to create a new table HDU for the data information
    dcol = fits.ColDefs(hdu[1].data)
    tbl2 = fits.BinTableHDU.from_columns(dcol)
    # Set the EXTNAME keyword to LDAC_OBJECTS
    tbl2.header["EXTNAME"] = "LDAC_OBJECTS"

    # Return the two LDAC format table HDUs
    return (tbl1, tbl2)


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
    # The paths of necessary files, including configuration file, input images, output files, and other parameters files
    fp_image = image_path
    fp_flag = image_flag_path
    fp_weight = image_weight_path
    fp_config = config_path
    fp_out = out_path
    fp_para = out_para_path
    fp_filter = filter_path

    # Construct the command line string for Source-Extractor
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

    # Execute the constructed Source-Extractor command
    os.system(cmd_sex)


def update_sext_catalog(sext_out_path: str, sext_out_path_up: str):
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
    # Open input catalog (star_info_SN_1.fits) to get posX_C1 and posY_C1
    directory = os.path.dirname(sext_out_path)

    # Construct the path to star_info_SN_1.fits
    star_info_path = os.path.join(directory, "star_info_SN_1.fits")

    # Read the image coodinate information from input catalog
    with fits.open(star_info_path) as star_info_hdulist:
        star_info_data = star_info_hdulist[1].data
        posX_C1 = star_info_data['posX_C1']
        posY_C1 = 9232 - star_info_data['posY_C1']

    # The paths of the Source-Extractor output and updated output
    fp_out = sext_out_path
    fp_out_up = sext_out_path_up

    # Filter data from the Source-Extractor output
    hdulist = fits.open(fp_out, mode="update")
    ptdata1 = hdulist[2].data  # LDAC
    ptdata1["FLUXERR_AUTO"] = ptdata1["FLUX_AUTO"] / 100

    # Match the data from the Source-Extractor output to the input catalog by celestial distance
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

    # Check the number of matched stars
    try:
        assert (len(tdata) > 30) and (len(tdata) < 30000)
    except AssertionError:
        raise AssertionError("too few stars or too much stars ")

    # Update the Source-Extractor output, only change the data part (hdulist[2].data).
    hdulist[2].data = tdata
    hdulist.flush()
    hdulist.writeto(fp_out_up, overwrite=True)

    # Return the filtered data
    return tdata


def write_reg(
    data,
    fp_reg: str,
    X_term: str,
    Y_term: str,
):
    """
    Write positions of filtered data to a .reg file.

    This function writes the coordinates of filtered data points to a .reg file,
    typically used for annotating regions or objects in an image.

    Parameters
    ----------
    data : fits.fitsrec.FITS_rec
        The filtered data from the Source-Extractor output. It should contain the X and Y coordinates.
    fp_reg : str
        The target .reg file path.
    X_term : str
        The column name representing the X coordinate in the data.
    Y_term : str
        The column name representing the Y coordinate in the data.

    Returns
    -------
    None

    Example
    -------
    >>> write_reg(tdata, filesatreg, "X_IMAGE", "Y_IMAGE")
    """
    # Open the file in append mode to write data
    with open(fp_reg, "a") as f:
        # Write the header information defining the shape style and selection status
        f.write(
            'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1"
            + "\n"
        )
        # Write the "physical" tag to specify pixel coordinates
        f.write("physical" + "\n")
        # Iterate through the data and write each point as a circle shape to the file
        for i in range(len(data)):
            # Format the X and Y coordinates and the circle radius, then write to the file
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

    This function calls the Scamp software to perform astrometry on astronomical images. It uses the output of Source-Extractor as input,
    references catalog, and generates a Scamp header file as output.

    Parameters
    ----------
    fp_sext_out : str
        The file path of the Source-Extractor output, which includes the positions and other information of detected sources.
    fp_refcat : str
        The file path of the reference catalog. Typically, this is accurate astrometric catalog like Gaia.
    fp_config_para : str
        The file path of the Scamp configuration file. This file specifies the parameters and settings for Scamp's astrometry process.
    fp_scamp_head : str
        The file path of the Scamp header output. This contains the astrometric solutions applied to the original images.
    fp_up_head : str
        The aheader suffix file.

    Returns
    -------
    None

    Examples
    -------
    >>> work_scamp(fp_sext_out, fp_refcat, fp_config_para, fp_scamp_head, fp_up_head)
    """
    # Construct the command line call for Scamp, including Source-Extractor output, config parameters and reference catalog
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
    # Execute the Scamp command
    os.system(cmd_scamp)

    # Check for the existence of Scamp's output header file
    try:
        assert os.access(fp_scamp_head, os.F_OK)
    except AssertionError:
        raise AssertionError("no Scamp result head file")


def rewrite_scamp_head_to_standard(scamp_head_path: str, scamp_head_bk_path: str):
    """
    Rewrite the Scamp header to the standard FITS format.

    This function modifies the Scamp header file to conform to the standard FITS header format,
    ensuring that each header block is 2880 bytes long, as per the FITS standard requirements.

    Parameters
    ----------
    scamp_head_path : str
        The path to the Scamp header file.
    scamp_head_bk_path : str
        The path where the rewritten Scamp header will be saved.

    Returns
    -------
    None

    Examples
    --------
    >>> rewrite_scamp_head_to_standard(fp_head_scamp, fp_head_bk_scamp)
    """
    fp_head = scamp_head_path
    fp_head_bk = scamp_head_bk_path

    # Check if the file path ends with ".head"
    if os.path.splitext(fp_head)[-1] == ".head":
        # Open the original file for reading and backup file for writing
        with open(fp_head, "r") as input_file, open(fp_head_bk, "w") as backup_file:
            # Initialize an empty string to store the modified content
            content = ""
            # Iterate through each line of the input file
            for i, line in enumerate(input_file.readlines(), start=1):
                line_tail = ""  # Line tail padding
                blank_line = ""  # Blank lines
                # If the line length is less than or equal to 81 characters, pad it with spaces
                if len(line) <= 81:
                    line_tail = " " * (81 - len(line))
                # Replace lines containing "Sorbonne" with a copyright comment
                if "Sorbonne" in line:
                    line = "COMMENT   (c) 2010-2018 Sorbonne Universite/Universite de Bordeaux/CNRS"
                # If the line contains "END" and it's the last line, add padding for alignment
                if "END" in line and i == len(input_file):
                    blank_line = " " * 80 * (36 - i % 36)
                    i = i + (36 - i % 36)
                # Concatenate the modified line, tail padding, and blank line to the content
                content += line + line_tail + blank_line
            # Write the modified content to the backup file, removing newline characters (ecah line's size changes from 81 to 80)
            # Therefore, the content's size will be multiples of 2880(80 * 36)
            backup_file.write(content.replace("\n", ""))


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
    ra_offset_median = dec_offset_median = 99999
    ra_offset_std = dec_offset_std = 99999
    match_num = 0

    mask = abs(refcat_now["X_WORLD"]) > 0  # and (abs( refdata['Y_WORLD'] >0 ))
    gaia_ra = refcat_now[mask]["X_WORLD"]
    gaia_dec = refcat_now[mask]["Y_WORLD"]

    ref_catalog = SkyCoord(ra=gaia_ra, dec=gaia_dec)
    obs_catalog = w.pixel_to_world(stardata["XWIN_IMAGE"] - 1, stardata["YWIN_IMAGE"] - 1)

    ref_ids, refd2obsd, _ = obs_catalog.match_to_catalog_sky(ref_catalog)

    ref_uids = np.unique(ref_ids)
    obs_uids = np.full_like(ref_uids, -1)

    for i, ref_uid in enumerate(ref_uids):
        ref_ids_filter = ref_ids == ref_uid
        refd2obsd_filter = refd2obsd[ref_ids_filter] == np.min(refd2obsd[ref_ids_filter])
        obs_uids[i] = np.where(ref_ids_filter)[0][refd2obsd_filter][0]

    refd2obsd_uid_lim = refd2obsd[obs_uids].arcsecond < 1

    if refd2obsd_uid_lim.sum() > 0:
        obs_uids_lim = obs_uids[refd2obsd_uid_lim]
        ref_uids_lim = ref_uids[refd2obsd_uid_lim]

        ref_ra_unique = ref_catalog[ref_uids_lim].ra.arcsec * np.cos(obs_catalog[obs_uids_lim].dec.deg * np.pi / 180)
        obs_ra_unique = obs_catalog[obs_uids_lim].ra.arcsec * np.cos(obs_catalog[obs_uids_lim].dec.deg * np.pi / 180)

        ref_dec_unique = ref_catalog[ref_uids_lim].dec.arcsec
        obs_dec_unique = obs_catalog[obs_uids_lim].dec.arcsec

        clip_ra_offset = sigma_clip((ref_ra_unique - obs_ra_unique), sigma=3)
        ra_offset_median = np.median(clip_ra_offset.data[~clip_ra_offset.mask])
        ra_offset_std = np.std(clip_ra_offset.data[~clip_ra_offset.mask])

        clip_dec_offset = sigma_clip(ref_dec_unique - obs_dec_unique, sigma=3)
        dec_offset_median = np.median(clip_dec_offset.data[~clip_dec_offset.mask])
        dec_offset_std = np.std(clip_dec_offset.data[~clip_dec_offset.mask])

        match_num = len(refd2obsd_uid_lim)

    return ra_offset_median, dec_offset_median, ra_offset_std, dec_offset_std, match_num


def update_standard_output_head(
    ast_head_path: str,
    meanra: float,
    meande: float,
    stdra: float,
    stdde: float,
    match_num: int,
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
    Copy img and modify it's header.

    Copy the file at fp_img to a new location, modify its header, and return the full path of the file at the new location.

    Parameters
    ----------
    fp_img : str
        Path to the image file.
    new_header : dict
        New header information to be updated in the copied file.

    Returns
    -------
    new_file_path
        Full path of the copied file at the new location.

    Examples
    --------
    >>> copy_and_modify_header(fp_img, headerkey)
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

    fp_img = copy_and_modify_header(fp_img, headerkey)
    fp_img_flag = img_flag_path
    fp_img_weight = img_weight_path

    fp_head_ast = ast_head_path

    with fits.open(fp_img) as hdul:
        print(hdul[1].header['CD2_2'])
    # initial_standard_output_head(fp_img, fp_head_ast)

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
