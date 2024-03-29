# %%
# Load
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

# import basis as b
import grid as g
import elliptic as ell
import variables as v

# %% 
# Set up grid
order = 4
res_x, res_y, res_u, res_v = 20, 20, 10, 10

# Build grid
print('Initializing grid...')
orders = np.array([order, order, order])
print('Grid initialized.')

lows = np.array([-np.pi, -np.pi, -5, -5])
highs = np.array([np.pi, np.pi, 5, 5])
elements = np.array([res_x, res_y, res_u, res_v])

grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, orders=orders)

# x_grid = g.FiniteSpaceGrid(low=-np.pi, high=np.pi, elements=9, order=2)
# print('The element spacing is {:.3e}'.format(x_grid.dx))

# Charge density (initial: just ones)
space_var = v.SpaceScalar1P(resolutions=[res_x, res_y])
# space var.arr_nodal =  # cp.ones((res_x, res_y, order))
# space_var.arr_nodal = cp.sin(np.pi * grid.x.device_arr)[:, None, None] * cp.ones_like(grid.y.device_arr) 
# space_var.arr_nodal = cp.ones_like(grid.x.device_arr)[:, None, None] * cp.sin(grid.y.device_arr)[None, :, :]
space_var.arr_nodal = cp.sin(grid.x.device_arr)[:, None, None] * cp.sin(grid.y.device_arr)[None, :, :]
space_var.fourier_transform()

# source = np.sin(2.0 * np.pi / x_grid.length * x_grid.arr)

XX, YY = np.meshgrid(grid.x.arr.flatten(), grid.y.arr.flatten(), indexing='ij')

KX, KY = np.meshgrid(grid.x.wavenumbers, grid.y.arr.flatten(), indexing='ij')

plt.figure()
plt.contourf(XX, YY, space_var.arr_nodal.reshape(grid.x.elements, grid.y.elements*grid.y.order).get(),
             cp.linspace(cp.amin(space_var.arr_nodal), cp.amax(space_var.arr_nodal), num=100).get())
plt.colorbar(ax=plt.gca())

spectral_abs = cp.abs(space_var.arr_spectral.reshape(grid.x.wavenumbers.shape[0], grid.y.elements*grid.y.order))

plt.figure()
plt.contourf(KX, KY, spectral_abs.get(),
             cp.linspace(cp.amin(spectral_abs), cp.amax(spectral_abs), num=100).get())
plt.colorbar(ax=plt.gca())

space_var.inverse_fourier_transform()
plt.figure()
plt.contourf(XX, YY, space_var.arr_nodal.reshape(grid.x.elements, grid.y.elements*grid.y.order).get(),
             cp.linspace(cp.amin(space_var.arr_nodal), cp.amax(space_var.arr_nodal), num=100).get())
plt.colorbar(ax=plt.gca())

plt.show()

# %%
# Set up elliptic operator
e = ell.Elliptic(poisson_coefficient=1)
# e.build_central_flux_operator(grid=x_grid, basis=x_grid.local_basis)
e.build_central_flux_operator_dirichlet_fourier(grid=grid.y, basis=grid.y.local_basis, wavenumbers=grid.x.wavenumbers)
e.invert_with_fourier(wavenumbers=grid.x.wavenumbers)

# %%
# Prepare solution
source = space_var.arr_spectral.get()
solution = np.zeros_like(source).reshape(source.shape[0], source.shape[1]*source.shape[2]) + 0j

rhs = cp.zeros((source.shape[0], source.shape[1]*source.shape[2] + 1)) + 0j
for idx in range(source.shape[0]):
    rhs[idx, :-1] = cp.einsum('jk,kn->jn', source[idx, :], grid.y.local_basis.device_mass).reshape(source.shape[1] * 
                                                                                                    source.shape[2])
    solution[idx, :] = (cp.matmul(e.inv_op[idx, :, :], rhs[idx, :])[:-1]).get() #  / (grid.y.J[0] ** 2)).get()

# %%
# Reshape and look at solution
solution_var = v.SpaceScalar1P(resolutions=[res_x, res_y])
reshape_solution = solution.reshape(space_var.arr_spectral.shape[0], 
                                    space_var.arr_spectral.shape[1], 
                                    space_var.arr_spectral.shape[2])
solution_var.arr_spectral = cp.asarray(reshape_solution)
solution_var.inverse_fourier_transform()
# nodal_solution = cp.fft.irfft(cp.fft.fftshift(reshape_solution, axes=0), axis=0)
# nodal_solution = cp.fft.irfft(reshape_solution), axis=0, norm='forward')
# print(nodal_solution.shape)
nodal_solution = solution_var.arr_nodal

# %%
plt.figure()
plt.contourf(XX, YY, nodal_solution.reshape(grid.x.elements, grid.y.elements*grid.y.order).get(),
             cp.linspace(cp.amin(nodal_solution), cp.amax(nodal_solution), num=100).get())
plt.colorbar(ax=plt.gca())
plt.show()

# %%
