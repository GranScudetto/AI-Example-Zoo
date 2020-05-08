"""
Utils related to different file operations

Creation Date: April 2020
Creator: Mafuba09
"""
# imports...
import os
import time

EXPERIMENTS_FOLDER = 'experiments'


def get_experiment_dir(script_file_path: str, sub_name: str = '') -> str:
    """
    Creates a result structure in the main directory for the corresponding experiment. For traceability the name is
    derived from the task name f.e. cifar for cifar10.py and adds the current time stamp to distinguish among multiple
    experiment runs.

    :param script_file_path: path of the calling script, used to derive task name
    :param sub_name: optional string to describe experiment
    :return: experiment directory
    """
    tm_struct = time.gmtime()  # gets current time
    script_filename = os.path.splitext(os.path.split(script_file_path)[-1])[0]  # removes .py from calling script
    # combine time-string and script name into experiment name
    out_dir = '.' + os.sep + EXPERIMENTS_FOLDER \
              + os.sep + sub_name + '{}_{}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(script_filename, tm_struct.tm_year,
                                                                                  tm_struct.tm_mon, tm_struct.tm_mday,
                                                                                  tm_struct.tm_hour, tm_struct.tm_min,
                                                                                  tm_struct.tm_sec)
    if not os.path.exists(out_dir):  # if directory is not existing
        os.makedirs(out_dir)  # create output directory

    return out_dir  # return path to output directory


def get_latest_experiment_dir(file_path=None) -> str:
    """
    Get latest experiment folder.

    :param file_path: Custom filepath to folder with *.h5 file
    :return: file path to latest experiment directory
    """

    # Check if experiment folder exists
    experiment_folder = os.path.join('.', EXPERIMENTS_FOLDER)
    if file_path is not None and os.path.exists(file_path):
        return file_path

    print('Checking experiment folder!')
    if os.path.exists(experiment_folder):
        experiment_dirs = os.listdir(experiment_folder)
        experiment_dirs.sort(reverse=True) # Sort highest to lowest
        if len(experiment_dirs) <= 0:
            return ''
    else:
        return ''

    return '.' + os.sep + EXPERIMENTS_FOLDER + os.sep + str(experiment_dirs[0])
