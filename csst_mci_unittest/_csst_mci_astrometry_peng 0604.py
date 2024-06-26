"""
Identifier:     csstXXXXX/csst_mci_lX_XXXXXX.py
Name:           csst_mci_lX_XXXXX.py
Description:    Test demo function.
Author:         Xiyan Peng
Created:        2023-11-14
Modified-History:
    2023-11-14, Xiyan Peng, created
    2023-11-29, Xiyan Peng, modified
"""
import os
from csst_mci_common.common import CsstMCIInit
from csst_mci_common.status import CsstStatus, CsstResult
from csst_mci_common.interface import BasePipelineInterface

csst_mci = CsstMCIInit()


class PipelineL1MCIAstrometry(BasePipelineInterface):
    """
    The class of MCI astrometry pipeline.

    The class includes three method: __init__(), run() and para_set(). The para_set()
    sets parameters of files for pipeline, and the run() runs the MCI astrometry pipeline
    and return running status.

    Attributes
    ----------
    obsid : str
        The obsid of this item.

    logger : csst_mci.logger
        Inheritance the Logging package for Python. Get a logger for CSST L1 Pipeline.

    status : CsstStatus
        Instance object of class CsstStatus of CSST L1 Pipeline. Get status.

    result : CsstResult
        Instance object of class CsstResult of CSST L1 Pipeline. Get result for the Pipeline.

    Methods
    -------
    para_set() -> CsstResult
        set parameters for L0/L1 pipeline.

    run() -> CsstResult
        Run the pipeline.

    Examples
    --------
    >>> case_astrometry = PipelineL1MCIAstrometry()
    >>> case_astrometry.para_set()
    >>> case_astrometry.run()
    """

    def __init__(self):
        """
        __init__ method.

        Initial method for logging, result, status and obsid.

        Returns
        -------
        None

        Notes
        -----
        Do not include the `self` parameter in the ``Parameters`` section.

        Examples
        --------
        >>> case_astrometry = PipelineL1MCIAstrometry()
        """
        self.result = CsstResult
        self.logger = csst_mci.logger
        self.status = CsstStatus

        self.obsid = None
        """str: Docstring *after* attribute, with type specified."""


    def run(self) -> CsstResult:
        """
        Method, run the pipeline.

        Run the MCI astrometry pipeline and return running status.

        Returns
        -------
        CsstResult
            Result containing `status`, `file_list`, and `output`.

        Notes
        -----
        Do not include the `self` parameter in the ``Parameters`` section.

        Examples
        --------
        >>> result = case_astrometry.run()
        """
        from . import run_mci_astrometry

        dm = self._dm

        # self.logger.info("%s running..." % self.__name__)
        self.status = run_mci_astrometry(dm=dm, logger=None)
        if self.status == 0:
            self.status = CsstStatus.PERFECT


        result = CsstResult(status=self.status, files=[])
        return result

    def para_set(self) -> CsstResult:
        """
        Method, set parameters of files for pipeline.

        Set parameters of files for pipeline, and return running status.

        Returns
        -------
        CsstResult
            Result containing `status`, `file_list`, and `output`.

        Notes
        -----
        Do not include the `self` parameter in the ``Parameters`` section.

        Examples
        --------
        >>> result = case_astrometry.para_set()
        """
        from csst_mci_common.data_manager import CsstMCIDataManager

        dm = csst_mci.dm

        obsid = "20100000001"
        detector = "C1"
        dir_ins = "/nfsdata/share/pipeline-unittest/csst_mci/csst_mci_instrument"
        dir_ast = dir_ins.replace("csst_mci_instrument", "csst_mci_astrometry")

        l1_proc_output = os.path.join(dir_ast, "l1_proc_output")
        ins_output = dir_ins
        sci_c1_path = sci_path = "/nfsdata/share/pipeline-unittest/csst_mci/data/csst_mci_instrument/CSST_MCI_C1_EXDF_20240419024541_20240419025041_20100000001_07_L1_V01.fits"
        img_path = os.path.join(ins_output, os.path.basename(sci_path.replace(".fits", "_img.fits")))
        wht_path = os.path.join(ins_output, os.path.basename(sci_path.replace(".fits", "_wht.fits")))
        head_path = os.path.join(ins_output, os.path.basename(sci_path.replace(".fits", "_img.head")))
        flag_path = os.path.join(ins_output, os.path.basename(sci_path.replace(".fits", "_flg.fits")))

        csst_mci.config.set_mci_config("obsid", obsid)
        csst_mci.config.set_mci_config("target_detector", detector)
        csst_mci.config.set_mci_config("l1_proc_output", l1_proc_output)
        csst_mci.config.set_mci_config("ins_output", ins_output)
        csst_mci.config.set_mci_config("sci_c1_path", sci_c1_path)
        csst_mci.config.set_mci_config("sci_path", sci_c1_path)
        csst_mci.config.set_mci_config("img_path", img_path)
        csst_mci.config.set_mci_config("wht_path", wht_path)
        csst_mci.config.set_mci_config("head_path", head_path)
        csst_mci.config.set_mci_config("flag_path", flag_path)

        self._dm = dm

        status = CsstStatus.PERFECT

        result = CsstResult(
            status=status,
        )
        return result

