from distutils.log import error
import numpy as np

def dente(x,d,u):
    y = np.array([np.heaviside(x.all()-d, 0.5)])
    ri = y.shape[0] 
    try:
        co = y.shape[1] 
    except:
        co = 1

    
    if ri == 1:
        ugo = co
    else:
        ugo = ri
    

    for i in range(0,ugo):
        if x[i] == d:
            if d == 0:
                y[i] = 0
            else:
                y[i] = 1
            
        if x[i] > u:
            y[i] = 0
    return y
