"""
NRV-I/O File Handler.
"""

import json
import os
import numpy as np


#################
# Miscalleneous #
#################
def is_iterable(some_stuff):
    """
    this function chels wether or not a variable contains an iterrable

    Parameters
    ----------
    some_stuff  :
        variable to check

    Returns
    -------
    False if a string or a number, True if iterrable (table, dict, tupple, numpy array...)
    """
    try:
        _ = (a for a in some_stuff)
        if isinstance(some_stuff, str):
            flag = False
        else:
            flag = True
    except TypeError:
        flag = False
    return flag


def rmv_ext(fname):
    """
    return filename without extension

    Parameters
    ----------
    fname   : str
        file name with or without extention

    Returns
    -------
    fname   : str
        file name without extention
    """
    if isinstance(fname, str):
        i = fname.rfind(".")
        if i > 0:
            fname = fname[:i]
    return fname


def generate_new_fname(fname):
    """
    Prevent overwriting existing files.
    if the filename exists, add a number or add one to the number at the end of
    the filename

    Parameters
    ----------
    fname : str
        name of the file to check
    """
    if os.path.isfile(fname):
        for i in range(len(fname)):
            if fname[-i - 1] == ".":
                try:
                    fname = (
                        fname[: -i - 2] + str(1 + int(fname[-i - 2])) + fname[-i - 1 :]
                    )
                except:
                    fname = fname[: -i - 1] + "0" + fname[-i - 1 :]
        return generate_new_fname(fname)
    return fname


#####################################
## Folder and archive related code ##
#####################################
def create_folder(foldername, access_rights=0o755):
    """
    create a folder with controled access rights.

    Parameters
    ----------
    foldername : str
        name of the folder to create
    access_rights : int
        unix like rights
    """
    try:
        os.mkdir(foldername, access_rights)
    except OSError:
        print(
            "Creation of the directory %s failed, this folder may already exist"
            % foldername
        )


#######################
## JSON related code ##
#######################
def check_json_fname(fname):
    """
    Add ".json" extension is missing at the end of the file name and check if it exists.

    Parameters
    ----------
    fname    : str
        name of the file

    Retruns
    -------
    fname    : str
        name of the file with the ".json" extension added if required

    Errors
    ------
    NRV_Error
        rised if fname does not exist
    """
    if fname[-5:] != ".json":
        fname += ".json"
    if os.path.isfile(fname):
        return fname
    else:
        print(fname + " not found cannot be load")


def json_dump(results, filename):
    """
    save stuff as a json file

    Parameters
    ----------
    results     :
        stuff to save
    filename    : str
        name of the file where results are saved
    """
    with open(filename, "w") as file_to_save:
        json.dump(results, file_to_save, cls=Encoder)


def json_load(filename):
    """
    Load stuff from a json file

    Parameters
    ----------
    filename    : str
        name of the file where results are stored

    Returns
    -------
    results : dictionary
        stuff from file
    """
    with open(check_json_fname(filename), "r") as file_to_read:
        results = json.load(file_to_read)
    return results


class Encoder(json.JSONEncoder):
    """
    Json encoding class,
    prevents from type error due to np.arrays
    solution taken as this from askpython.com
    """

    def default(self, obj):
        # If the object is a numpy array
        if isinstance(obj, np.integer):
            result = int(obj)
        elif isinstance(obj, np.floating):
            result = float(obj)
        elif isinstance(obj, np.ndarray):
            result = obj.tolist()
        else:
            # Let the base class Encoder handle the object
            result = json.JSONEncoder.default(self, obj)
        return result
