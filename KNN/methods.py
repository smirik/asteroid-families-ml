import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix


ALLOWED_METHODS = (KNeighborsClassifier, GradientBoostingClassifier)


def zappala_metric(x, y, k1 = 1.25, k2 = 2.0, k3 = 2.0):
    '''
    Realization of a distance function defined by Zappala et al.
    See e.g. http://adsabs.harvard.edu/abs/1990AJ....100.2030Z
    
    "x" and "y" are the vectors in the (a, e, sinI, n) phase space.
    '''

    return ((x[0] + y[0]) * (x[3] + y[3]) / 4.0 *
            np.sqrt(
            4.0 * k1 * ((x[0] - y[0]) / (x[0] + y[0]))**2 +
            k2 * (x[1] - y[1])**2 +
            k3 * (x[2] - y[2])**2)
            )

