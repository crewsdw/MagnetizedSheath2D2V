import numpy as np
import cupy as cp

import basis as b
import grid as g
import elliptic as ell
import variables as v

import matplotlib.pyplot as plt
import matplotlib

# RC 
font = {'size': 18}
matplotlib.rc('font', **font)

# Parameters
order = 9
# time_order = 3

res_x, res_y, res_u, res_v = 10, 10, 40, 40

# Build grid
print('Initializing grid...')
orders = np.array([order, order, order])
print('Grid initialized.')

lows = np.array([-1, -1, -5, -5])
highs = np.array([1, 1, 5, 5])
elements = np.array([res_x, res_y, res_u, res_v])

grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, orders=orders)

# Build elliptic operator
print('\nInitializing elliptic operator...')
e = ell.Elliptic(poisson_coefficient=1)
e.build_central_flux_operator(grid=grid.y, basis=grid.y.local_basis)
e.invert(wavenumbers=grid.x.wavenumbers)

print(e.inv_op.shape)

# Charge density (initial: just ones)
space_var = v.SpaceScalar1P(resolutions=[res_x, res_y])
# space var.arr_nodal =  # cp.ones((res_x, res_y, order))
# space_var.arr_nodal = cp.sin(np.pi * grid.x.device_arr)[:, None, None] * cp.ones_like(grid.y.device_arr) 
space_var.arr_nodal = cp.ones_like(grid.x.device_arr)[:, None, None] * cp.sin(np.pi * grid.y.device_arr)[None, :, :]
space_var.fourier_transform()

plt.figure()
plt.contourf(space_var.arr_nodal.reshape(grid.x.elements, grid.y.elements*grid.y.order).get())

plt.figure()
plt.plot(grid.y.arr.flatten(), space_var.arr_nodal[0, :, :].flatten().get(), 'o--')
plt.grid(True)
plt.show()

rhs = cp.zeros((space_var.arr_nodal.shape[0], space_var.arr_nodal.shape[1]*space_var.arr_nodal.shape[2] + 1))
# rhs[:, :-1] = cp.tensordot(space_var.arr_nodal, grid.y.local_basis.device_mass, axes=([2], [1])).reshape(elements[0], elements[1]*order)
rhs[:, :-1] = cp.einsum('ijk,kn->ijn', space_var.arr_nodal, grid.y.local_basis.device_mass).reshape(elements[0], elements[1]*order)

print(rhs.shape)

# solution = cp.einsum('jk,j->k', e.inv_op[0,:,:], rhs[0,:])[:-1] / (grid.y.J[0] ** 2)
solution = cp.matmul(e.inv_op[0,:,:], rhs[0,:]) / (grid.y.J[0]**2)
print(solution[-1])
solution = solution[:-1]
plt.figure()
plt.plot(grid.y.arr.flatten(), solution.get())
plt.plot(grid.y.arr.flatten(), -1.0*np.sin(np.pi * grid.y.arr.flatten())/np.pi**2, 'k')
plt.grid(True)
plt.show()

quit()

print(space_var.arr_spectral.shape)

# Solve the equation
# solution = 
reshaped_var = space_var.arr_spectral.reshape((space_var.arr_spectral.shape[0], 
                                              space_var.arr_spectral.shape[1] * space_var.arr_spectral.shape[2]))

reshape_shape = (space_var.arr_spectral.shape[0], 
                                              space_var.arr_spectral.shape[1] * space_var.arr_spectral.shape[2])

# print(reshaped_var.shape)
# reshape_with_gauge = cp.zeros((reshaped_var.shape[0], reshaped_var.shape[1] + 1)) + 0j
# reshape_with_gauge[:, :-1] = reshaped_var

# print(reshape_with_gauge.shape)

# Preprocess (last entry is average value)
rhs = cp.zeros((reshaped_var.shape[0], reshaped_var.shape[1] + 1)) + 0j
rhs[:, :-1] = cp.tensordot(space_var.arr_spectral, grid.y.local_basis.device_mass, axes=([2], [1])).reshape(reshape_shape)
# rhs[:, :-1] = space_var.arr_spectral.reshape(reshape_shape)

solution = cp.einsum('ikj, ik->ij', e.inv_op, rhs)[:, :-1]
print(solution.shape)

reshape_solution = solution.reshape(6, 10, order)
nodal_solution = cp.fft.irfft(cp.fft.fftshift(reshape_solution, axes=0), axis=0)
print(nodal_solution.shape)

nodal_sol_plot = nodal_solution.reshape((10, 40))

plt.figure()
plt.contourf(nodal_sol_plot.get())
plt.colorbar()

plt.figure()
plt.plot(grid.y.arr.flatten(), nodal_sol_plot[0, :].get(), 'o--')
plt.show()
