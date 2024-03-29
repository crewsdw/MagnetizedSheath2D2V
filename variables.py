import cupy as cp
import numpy as np
# import dispersion
import scipy.optimize as opt


class SpaceScalar1P:
    """ Class for configuration-space scalar scalars with one periodic and one non-periodic direction """
    def __init__(self, resolutions):
        self.res_x, self.res_y = resolutions
        self.arr_nodal, self.arr_spectral = None, None
    
    def fourier_transform(self):
        # self.arr_spectral = cp.fft.fftshift(cp.fft.rfft(self.arr_nodal, norm='forward', axis=0), axes=(0,))
        self.arr_spectral = cp.fft.rfft(self.arr_nodal, norm='forward', axis=0)
    
    def inverse_fourier_transform(self):
        # self.arr_nodal = cp.fft.irfft(cp.fft.fftshift(self.arr_spectral, axes=(0,)), norm='forward', axis=0)
        self.arr_nodal = cp.fft.irfft(self.arr_spectral, norm='forward', axis=0)

    def integrate(self, grid):  # , array
        """ Integrate nodal array """
        # arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        # Reduce nodal array on non-periodic direction's quadrature points
        array = grid.integrate(self.arr_nodal, idx=[1,2])  # idx = [1,2] or just 1 ?
        arr_add = cp.zeros((self.res_x + 1))
        arr_add[:-1, :-1] = array
        arr_add[-1, :-1] = array[0, :]
        arr_add[:-1, -1] = array[:, 0]
        arr_add[-1, -1] = array[0, 0]
        return trapz1D(arr_add, grid.x.dx)

class SpaceScalar2P:
    """ Class for configuration-space scalars with two periodic directions """

    def __init__(self, resolutions):
        self.res_x, self.res_y = resolutions
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.rfft2(self.arr_nodal, norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.fft.irfft2(cp.fft.fftshift(self.arr_spectral, axes=0), norm='forward')

    def integrate(self, grid, array):
        """ Integrate an array, possibly self """
        # arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        arr_add = cp.zeros((self.res_x + 1, self.res_y + 1))
        arr_add[:-1, :-1] = array
        arr_add[-1, :-1] = array[0, :]
        arr_add[:-1, -1] = array[:, 0]
        arr_add[-1, -1] = array[0, 0]
        return trapz2D(arr_add, grid.x.dx, grid.y.dx)

    def integrate_energy(self, grid):
        self.integrate(grid=grid, array=0.5 * self.arr_nodal ** 2.0)

def trapz1D(f, dx):
    """ Custom trapz1D routine, sum  """
    return cp.sum(f[:-1] + f[1:], axis=1) * dx / 2.0

def trapz2D(f, dx, dy):
    """ Custom trapz2D routine using cupy """
    sum_y = cp.sum(f[:, :-1] + f[:, 1:], axis=1) * dy / 2.0
    return cp.sum(sum_y[:-1] + sum_y[1:]) * dx / 2.0
