# %%
# Load
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

# import basis as b
import grid as g
import elliptic as ell
# import variables as v

# %% 
# Set up grid
x_grid = g.FiniteSpaceGrid(low=-np.pi, high=np.pi, elements=9, order=2)
print('The element spacing is {:.3e}'.format(x_grid.dx))

source = np.sin(2.0 * np.pi / x_grid.length * x_grid.arr)

plt.figure()
plt.plot(x_grid.arr.flatten(), source.flatten())
plt.grid(True)
plt.show()

# %%
# Set up elliptic operator
e = ell.Elliptic(poisson_coefficient=1)
# e.build_central_flux_operator(grid=x_grid, basis=x_grid.local_basis)
e.build_central_flux_operator_dirichlet(grid=x_grid, basis=x_grid.local_basis)
e.invert()

# %%
# Prepare solution
rhs = cp.zeros((source.shape[0]*source.shape[1] + 1))
rhs[:-1] = cp.einsum('jk,kn->jn', source, x_grid.local_basis.device_mass).reshape(source.shape[0] * 
                                                                                       source.shape[1])

solution = cp.matmul(e.inv_op, rhs) / (x_grid.J[0] ** 2)

# %%
# Examine solution

plt.figure()
plt.plot(x_grid.arr.flatten(), -1.0 * source.flatten(), label='Exact solution')
plt.plot(x_grid.arr.flatten(), solution[:-1].get(), 'o--', label='Numerical solution')
plt.legend(loc='best'), plt.grid(True)

error = (-1.0 * source.flatten() - solution[:-1].get()).reshape(source.shape[0], source.shape[1])
L2_error = np.sqrt(np.einsum('ik,ik->i', np.einsum('jk,ij->ik', x_grid.local_basis.mass, error), error).sum()) / x_grid.length

# Check quadratic integration with mass matrix (result: it works)
# sin_sqrd = np.einsum('ik,ik->i', np.einsum('jk,ij->ik', x_grid.local_basis.mass, source), source).sum() / x_grid.length
# print(sin_sqrd / x_grid.J[0].get())

print('The L2 error is {:.3e}'.format(L2_error))

# %%
# 
grad = cp.einsum('ijkl,kl->ij', e.gradient_operator, cp.asarray(source)).flatten() * x_grid.J[0]
error = grad.get() - np.cos(x_grid.arr.flatten())
L2_error = np.linalg.norm(error) / x_grid.length

print(L2_error)

plt.figure()
plt.plot(x_grid.arr.flatten(), grad.get(),
         'o--')
plt.plot(x_grid.arr.flatten(), np.cos(x_grid.arr.flatten()))

plt.figure()
plt.plot(x_grid.arr.flatten(), error, 'o--')
plt.xlabel('x'), plt.ylabel('Pointwise error')

plt.show()

# %%
# Automate it
orders = np.array([2, 3, 4, 5, 6, 7, 8])  # , 6, 7, 8, 9, 10])
Nx = np.array([3, 5, 8, 13])  # , 21, 34, 55, 89, 144, 233, 377, 610])

dx = np.zeros((orders.shape[0], Nx.shape[0]))
penalty = np.zeros_like(dx)
errors = np.zeros_like(dx)

for idx_o, order in enumerate(orders):
    for idx_n, N in enumerate(Nx):
        x_grid = g.FiniteSpaceGrid(low=-np.pi, high=np.pi, elements=N, order=order)

        source = np.sin(2.0 * np.pi / x_grid.length * x_grid.arr)

        e = ell.Elliptic(poisson_coefficient=1)
        e.build_central_flux_operator_dirichlet(grid=x_grid, basis=x_grid.local_basis)
        e.invert()
        
        rhs = cp.zeros((source.shape[0]*source.shape[1] + 1))
        rhs[:-1] = cp.einsum('jk,kn->jn', source, x_grid.local_basis.device_mass).reshape(
            source.shape[0] * source.shape[1])
        solution = cp.matmul(e.inv_op, rhs) / (x_grid.J[0] ** 2)
        error = (-1.0 * source.flatten() - solution[:-1].get()).reshape(source.shape[0], source.shape[1])
        L2_error = np.sqrt(np.einsum('ik,ik->i', np.einsum('jk,ij->ik', x_grid.local_basis.mass, error), error).sum()) / x_grid.length

        dx[idx_o, idx_n] = x_grid.dx
        penalty[idx_o, idx_n] = e.penalty
        errors[idx_o, idx_n] = L2_error

# print('The element spacing is {:.3e}'.format(x_grid.dx))

plt.figure()
for idx in range(orders.shape[0]):
    plt.loglog(dx[idx, :], errors[idx, :], label='order=' + str(orders[idx]))
plt.grid(True)
plt.legend(loc='best')

# %%
plt.figure()
for idx in range(orders.shape[0]):
    plt.loglog(dx[idx, :], errors[idx, :], 'o--', label='o='+str(orders[idx]))
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'Element spacing $\mathcal{h}$'), plt.ylabel('L2 error')
plt.title('Poisson solution errors and convergence'), plt.tight_layout()

plt.figure()
for idx in range(orders.shape[0]):
    plt.loglog(dx[idx, :], penalty[idx, :], label='o='+str(orders[idx]))

# %%
# slopes
slopes = np.zeros_like(orders) + 0.0
for idx in range(orders.shape[0]):
    slopes[idx] = (np.log10(errors[idx,3]) - np.log10(errors[idx,1])) / (np.log10(dx[idx,3])-np.log10(dx[idx,1]))

print(orders)
print(slopes)

# %%
# Test 