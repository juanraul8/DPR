'''
    define some helper functions for shtools
'''
import pyshtools
from pyshtools.expand import MakeGridDH
import numpy as np

def shtools_matrix2vec(SH_matrix):
    '''
        for the sh matrix created by sh tools, 
        we create the vector of the sh
    '''
    numOrder = SH_matrix.shape[1]
    vec_SH = np.zeros(numOrder**2)
    count = 0
    for i in range(numOrder):
        for j in range(i,0,-1):
            vec_SH[count] = SH_matrix[1,i,j]
            count = count + 1
        for j in range(0,i+1):
            vec_SH[count]= SH_matrix[0, i,j]
            count = count + 1
    return vec_SH

def shtools_sh2matrix(coefficients, degree):
    '''
        convert vector of sh to matrix
    '''
    coeffs_matrix = np.zeros((2, degree + 1, degree + 1))
    current_zero_index = 0
    for l in range(0, degree + 1):
        coeffs_matrix[0, l, 0] = coefficients[current_zero_index]
        for m in range(1, l + 1):
            coeffs_matrix[0, l, m] = coefficients[current_zero_index + m]
            coeffs_matrix[1, l, m] = coefficients[current_zero_index - m]
        current_zero_index += 2*(l+1)
    return coeffs_matrix 

def shtools_getSH(envMap, order=5):
    '''
        get SH based on the envmap
    '''
    SH =  pyshtools.expand.SHExpandDH(envMap, sampling=2, lmax_calc=order, norm=4)
    return SH

#---------------------------------------Juan Code------------------------------------------------------------
def shtools_dense():

    # Ring trajectory
    num_pts=200
    phi_in= 75

    phi = np.empty(num_pts)
    phi.fill(phi_in)

    theta = np.arange(0, 360, 360 / num_pts)

    #print (phi)
    #print (theta)

    #Compute SH coefficients
    coeffs = []

    for i in range(num_pts):
        mat = pyshtools.expand.spharm_functions.spharm(2, phi[i], theta[i])
        vec = shtools_matrix2vec(mat)

        #print (mat.shape)
        #print (mat)

        #print (vec.shape)
        #print (vec)
        #print ("##################")

        coeffs.append(vec)

    print (len(coeffs))

    return coeffs