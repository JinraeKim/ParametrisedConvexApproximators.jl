using PyCall

torch = pyimport("torch")
np = pyimport("numpy")

"""
Note: the below code is a minimal test of using `cvxpylayers` and `torch` to
incorporate "differentiable convex programming".
Unfortunately, differentiable convex programming seems not realised in pure Julia packages.
"""


"""
Test for cvxpylayers
"""
function main()
    py"""
    import cvxpy as cp
    import torch
    from cvxpylayers.torch import CvxpyLayer
    import torch.optim as optim
    import copy


    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    A_tch = torch.randn(m, n, requires_grad=True)
    b_tch = torch.randn(m, requires_grad=True)

    # solve the problem
    solution, = cvxpylayer(A_tch, b_tch)
    print(A_tch)
    A_tch_prev = copy.deepcopy(A_tch)

    # compute the gradient of the sum of the solution with respect to A, b
    optimiser = optim.SGD((A_tch, b_tch), lr=1e3)
    optimiser.zero_grad()
    solution.sum().backward()
    print(A_tch.grad)
    optimiser.step()
    print(A_tch)
    print(A_tch == A_tch_prev)
    """
end

"""
Test for network construction and inference
"""
function main2(n)
    py"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Linear($$n, 10)

        def forward(self, x):
            x = self.fc(x)
            return x

    """
    # a = torch.from_numpy(np.array([1, 2, 3, 4, 5.], dtype=np.float32))
    a = torch.from_numpy(PyObject(np.random.randn(n)).astype(np.float32))
    net = py"Net"()
    net(a)
end
