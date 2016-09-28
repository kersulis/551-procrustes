import matplotlib.pyplot as plt
import numpy as np

def plot_digits(X, Y, Ya):
    # plot X, Y, Ya on GCA axes
    plt.plot(X[:,0], X[:,1], '-o', fillstyle='none', label='X')
    plt.plot(Y[:,0], Y[:,1], '-kx', label='Y misaligned')
    plt.plot(Ya[:,0], Ya[:,1], '-v', fillstyle='none', label='Y aligned')
    plt.axis('equal')
    plt.legend(fontsize=9)

def procrustes(shapes):
    """
    Syntax:     ashapes, sm = procrustes(shapes)

    Inputs:     shapes is a n x p x d array, where n is the number of
                shapes, p is the number of points per shape, and d is
                the dimension of each point (d=2 for xy plane, etc.)

    Outputs:    ashapes contains data in the same format as shapes,
                where all shapes are aligned to the first. sm is the
                mean of these shapes.
    """
    ashapes = np.zeros(shapes.shape)
    sm = shapes[0,:,:].astype(float) # initialize mean shape
    sm -= sm.mean(0)
    ashapes[0,:,:] = sm

    for (i, s) in enumerate(shapes[1:,:,:]):
        sa = align(sm, s) # align s to mean shape sm
        ashapes[i+1,:,:] = sa
        sm = (sm + sa)/2

    return ashapes, sm

def align(s1, s2):
    """
    Syntax:     s2a = align(s1, s2)

    Inputs:     s1, s2 are p x d arrays, where p is the number of
                points per shape and d the dimension of each (d=2
                for xy plane, etc.)

    Outputs:    s2a, an aligned version of s2 translated and rotated
                to overlap s1
    """
    # Translate so centroids are 0
    s1c = s1.astype(float) - s1.mean(0)
    s2c = s2.astype(float) - s2.mean(0)

    # Align using SVD
    (U, _, Vt) = np.linalg.svd(np.dot(s1c.T, s2c))
    s2a = s2c.dot(Vt.T).dot(U.T)

    # Translate to match original centroid of X
    return s2a + s1.mean(0)

def plot_molecules(moleculeData):
    """
    Plot data for molecules on GCA axis. Make sure axis has 3D
    projection.
    """
    for m in moleculeData:
        x, y, z = m[:,0], m[:,1], m[:,2]
        plt.plot(x, y, z, '-x')
