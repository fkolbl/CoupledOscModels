from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
from numpy import iterable

from ._file_handler import json_dump, json_load
from ._log_interface import pass_debug_info

########################################
#           check object               #
########################################


def is_BaseClass(x):
    """
    Check if the object x is a ``BaseClass``.

    Parameters
    ----------
    x   : any
        object to check.

    Returns
    -------
    bool
    """
    return isinstance(x, BaseClass)


def is_BaseClass_list(x):
    """
    check if the object x is a list containing only ``BaseClass``.

    Parameters
    ----------
    x   : any
        object to check.

    Returns
    -------
    bool
    """
    if iterable(x):
        for xi in x:
            if not is_BaseClass(xi):
                return False
        return True
    return False


def is_BaseClass_dict(x):
    """
    check if the object x is a dictionary containing only ``BaseClass``.

    Parameters
    ----------
    x   : any
        object to check.

    Returns
    -------
    bool
    """
    if isinstance(x, dict):
        for xi in x.values():
            if not is_BaseClass(xi):
                return False
        return True
    return False


##########################################
#           check dictionaries           #
##########################################
def is_BaseClass_dict(x):
    """
    Check if the object x is a dictionary of saved ``BaseClass``.

    Parameters
    ----------
    x : any
        object to check.

    Returns
    -------
    bool
    """
    if isinstance(x, dict):
        if "nrv_type" in x:
            return True
    return False


def is_BaseClass_dict_list(x):
    """
    Check if the object x is a list of dictionary of saved ``BaseClass``.

    Parameters
    ----------
    x : any
        object to check.

    Returns
    -------
    bool
    """
    if iterable(x):
        if len(x) > 0:
            for xi in x:
                if not (is_BaseClass_dict(xi)):
                    return False
            return True
    return False


def is_BaseClass_dict_dict(x):
    """
    Check if the object x is a dictionary containing dictionaries of saved ``BaseClass``.

    Parameters
    ----------
    x : any
        object to check.

    Returns
    -------
    bool
    """
    if isinstance(x, dict):
        for key in x:
            if not (is_BaseClass_dict(x[key])):
                return False
        return True
    return False


def is_BaseObject_dict(x):
    """
    Check if the object x is_BaseClass_dict, is_BaseClass_dict_list or is_BaseClass_dict_dict.

    Parameters
    ----------
    x : any
        object to check.

    Returns
    -------
    bool
    """
    return is_BaseClass_dict(x) or is_BaseClass_dict_list(x) or is_BaseClass_dict_dict(x)


######################################
#       numpy compatibility          #
######################################
def is_empty_iterable(x):
    """
    check if the object x is an empty iterable

    Parameters
    ----------
    x : any
        object to check.

    Returns
    -------
    bool
    """
    if not np.iterable(x):
        return False
    if len(x) == 0:
        return True
    return False


######################################
#            NRV Class               #
######################################


class BaseClass(metaclass=ABCMeta):
    """
    Instanciate a basic class
    Base Class are empty shells, defined as abstract classes of which every class
    should inherite. This enable automatic context backup with save and load methods.
    """

    @abstractmethod
    def __init__(self):
        """
        Init method for ``BaseClass``
        """
        self.__BaseObject__ = True
        self.object_type = self.__class__.__name__
        pass_debug_info(self.object_type, " initialized")

    def __del__(self):
        """
        Destructor for ``BaseClass``
        """
        pass_debug_info(self.object_type, " deleted")
        keys = list(self.__dict__.keys())
        for key in keys:
            del self.__dict__[key]

    def save(self, save=False, fname="instance_save.json", blacklist=[], **kwargs) -> dict:
        """
        Generic saving method for ``BaseClass`` instance.

        Parameters
        ----------
        save: bool, optional
            If True, saves the object in a json file, by default False.
        fname : str, optional
            Name of the json file
        blacklist : dict, optional
            Dictionary containing the keys to be excluded from the saving.
        **kwargs : dict, optional
            Additional arguments to pass to the ``save`` method of the object.

        Returns
        -------
        key_dict : dict
            dictionary containing the original instance data in a `jsonisable` format.

        Note
        ----
        - This ``save`` method does not save the object to a `json` file by default: It only\
            returns a dictionary containing the original instance data in a jsonisable format.\
            However, this is the simplest way to do it by setting the ``save`` parameter to ``True``.
        - The dictionary returned by this `save` method can be modified without having any impact on the\
            the original instance (the items are deep copies of the instance's attributes).
        """
        key_dic = {}
        for key in self.__dict__:
            if key not in blacklist:
                if is_BaseClass(self.__dict__[key]):
                    key_dic[key] = self.__dict__[key].save(**kwargs)
                elif is_BaseClass_list(self.__dict__[key]):
                    key_dic[key] = []
                    for i in range(len(self.__dict__[key])):
                        key_dic[key] += [self.__dict__[key][i].save(**kwargs)]
                elif is_BaseClass_dict(self.__dict__[key]):
                    key_dic[key] = {}
                    for i in self.__dict__[key]:
                        key_dic[key][i] = self.__dict__[key][i].save(**kwargs)

                else:
                    key_dic[key] = deepcopy(self.__dict__[key])
        if save:
            json_dump(key_dic, fname)
        return key_dic

    def load(self, data, blacklist={}, **kwargs) -> None:
        """
        Generic loading method for ``BaseClass`` instance

        Parameters
        ----------
        data : str, dict, BaseClass
            data from which the object should be generated:

                - if str, data will be loaded from the corresponding json file
                - if dict, data will be loaded from a dictionnary
                - if BaseClass, same object will be returned
        blacklist : dict, optional
            Dictionary containing the keys to be excluded from the load
        **kwargs : dict, optional
            Additional arguments to be passed to the load method of the object
        """
        if isinstance(data, str):
            key_dic = json_load(data)
        else:
            key_dic = data
        for key in self.__dict__:
            if key in key_dic and key not in blacklist:
                if is_BaseObject_dict(key_dic[key]):
                    self.__dict__[key] = load_any(key_dic[key], **kwargs)
                elif isinstance(self.__dict__[key], np.ndarray):
                    self.__dict__[key] = np.array(key_dic[key])
                elif is_empty_iterable(key_dic[key]):
                    self.__dict__[key] = eval(self.__dict__[key].__class__.__name__)()
                else:
                    self.__dict__[key] = key_dic[key]

    def set_parameters(self, **kawrgs) -> None:
        """
        Generic method to set any attribute of ``BaseClass`` instance

        Parameters
        ----------
        ***kwargs
            Key arguments containing one or multiple parameters to set.

        """
        for key in kawrgs:
            if key in self.__dict__:
                self.__dict__[key] = kawrgs[key]

    def get_parameters(self):
        """
        Generic method returning all the atributes of an BaseClass instance

        Returns
        -------
            dict : dictionnary of all atributes of ``BaseClass`` instance
        """
        return self.__dict__