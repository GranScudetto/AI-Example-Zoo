"""
Utils related to different file operations

Creation Date: April 2020
Creator: Mafuba09
"""
# imports...
import os
import time


def get_experiment_dir(script_file_path) -> str:
    """
    Creates a result structure in the main directory for the corresponding experiment. For traceability the name is
    derived from the task name f.e. cifar for cifar10.py and adds the current time stamp to distinguish among multiple
    experiment runs.

    :param script_file_path: path of the calling script, used to derive task name
    :return: experiment directory
    """
    tm_struct = time.gmtime()  # gets current time
    script_filename = os.path.splitext(os.path.split(script_file_path)[-1])[0]  # removes .py from calling script
    # combine time-string and script name into experiment name
    out_dir = '.' + os.sep + 'experiments' + os.sep + '{}_{}{}{}_{}{}{}'.format(script_filename, tm_struct.tm_year,
                                                                                tm_struct.tm_mon, tm_struct.tm_mday,
                                                                                tm_struct.tm_hour, tm_struct.tm_min,
                                                                                tm_struct.tm_sec)
    if not os.path.exists(out_dir):  # if directory is not existing
        os.makedirs(out_dir)  # create output directory

    return out_dir  # return path to output directory
