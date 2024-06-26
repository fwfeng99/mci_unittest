# -*- coding: utf-8 -*-
# 下面的网页参考numpy风格注释说明
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
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


# 数据处理软件编写主调用类范例
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

        # 在此处编写代码，使得用户通过执行此方法运行程序

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
        # 可以在此处import需要的包，最好在文件头
        from . import run_mci_astrometry

        # 获取在para_set中配置的参数，此处用于整理，也可以不写，直接喂给需要运行的程序。
        dm = self._dm

        # 核心调用代码，用于调用开发者各自开发的程序，其中self.status是个人自定义的返回码，可以以任意形式返回
        # self.logger.info("%s running..." % self.__name__)
        self.status = run_mci_astrometry(dm=dm, logger=None)
        if self.status == 0:
            self.status = CsstStatus.PERFECT

        # 根据程序运行结果，配置返回成功、警告、或者报错状态
        # self.status = CsstStatus.PERFECT
        # print(type(self.status))
        # print(self.status)

        # 最终返回结果，需要严格按照如下格式

        result = CsstResult(status=self.status, files=[])
        return result

    # 在此处编写代码，使得用户通过执行此方法配置流水线运行参数，所有参数应当含有默认值，输入参数非必须。此处可以通过指定参数，给与不同情况的初始化参数配置
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
        dir_l0 = "/nfsdata/share/pipeline-unittest/csst_mci/data/dfs_dummy/L0/sim_ver20230725"
        dir_l1 = "/nfsdata/share/pipeline-unittest/csst_mci/data/dfs_dummy/L1/sim_ver20230725"
        total_root = dir_l1
        l0_file_root_path = dir_l0
        l1_root_output = os.path.join(total_root, obsid)
        l1_proc_output = os.path.join(l1_root_output, "l1_proc_output")
        ins_output = os.path.join(l1_root_output, "ins_output")
        ast_output = os.path.join(l1_root_output, "ast_output")
        l0_sci_c1_path = "/nfsdata/share/pipeline-unittest/csst_mci/data/dfs_dummy/L0/sim_ver20230725/20100000001/CSST_MCI_C1_EXDF_20230918003941_20230918004441_20100000001_07_L0_V01.fits"
        sci_path = l0_sci_c1_path.replace("_L0_", "_L1_")
        img_path = os.path.join(
            ins_output, os.path.basename(sci_path.replace(".fits", "_img.fits"))
        )
        wht_path = os.path.join(
            ins_output, os.path.basename(sci_path.replace(".fits", "_wht.fits"))
        )
        head_path = os.path.join(
            ins_output, os.path.basename(sci_path.replace(".fits", "_img.head"))
        )
        flag_path = os.path.join(
            ins_output, os.path.basename(sci_path.replace(".fits", "_flg.fits"))
        )
        ast_path = img_path.replace("_img.fits", "_ast.fits")

        csst_mci.config.set_mci_config("obsid", obsid)
        csst_mci.config.set_mci_config("target_detector", detector)
        csst_mci.config.set_mci_config("total_root", total_root)
        csst_mci.config.set_mci_config("l0_file_root_path", l0_file_root_path)
        csst_mci.config.set_mci_config("l1_root_output", l1_root_output)
        csst_mci.config.set_mci_config("l1_proc_output", l1_proc_output)
        csst_mci.config.set_mci_config("ins_output", ins_output)
        csst_mci.config.set_mci_config("sci_c1_path", l0_sci_c1_path)
        csst_mci.config.set_mci_config("sci_path", l0_sci_c1_path)
        csst_mci.config.set_mci_config("img_path", img_path)
        csst_mci.config.set_mci_config("wht_path", wht_path)
        csst_mci.config.set_mci_config("head_path", head_path)
        csst_mci.config.set_mci_config("flag_path", flag_path)
        csst_mci.config.set_mci_config("ast_path", ast_path)
        csst_mci.config.set_mci_config("ast_c1_path", ast_path)
        csst_mci.config.set_mci_config("ast_output", ast_output)

        self._dm = dm

        status = CsstStatus.PERFECT

        # 最终返回结果
        result = CsstResult(
            status=status,
        )
        return result
