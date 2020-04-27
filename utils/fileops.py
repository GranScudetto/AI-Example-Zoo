import os
import time

# Time
def get_output_dir(script_file_path) -> str:
    tm_struct = time.gmtime()
    script_filename = os.path.splitext(os.path.split(script_file_path)[-1])[0]
    out_dir = '.' + os.sep + 'Experiment' + os.sep + '{}_{}{}{}_{}{}{}'.format(script_filename,
                                                                              tm_struct.tm_year,
                                                                              tm_struct.tm_mon,
                                                                              tm_struct.tm_mday,
                                                                              tm_struct.tm_hour,
                                                                              tm_struct.tm_min,
                                                                              tm_struct.tm_sec)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir