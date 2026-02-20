import numpy as np

def norm_database(mu,sigma,N):
    mu0 = mu[0]
    mu1 = mu[1]
    sigma0 = sigma[0]
    sigma1 = sigma[1]
    col = len(mu0)

    class0 = np.random.normal(mu0[0], sigma0[0],[N,1])
    class1 = np.random.normal(mu1[0],sigma1[0],[N,1])

    for i in range(1,col):
        v0 = np.random.normal(mu0[i],sigma0[i],[N,1])
        class0 = np.hstack((class0,v0))

        v1 = np.random.normal(mu1[i],sigma1[i],[N,1])
        class1 = np.hstack((class1,v1))

    Y1 = np.ones((N,1),dtype=bool)
    Y0 = np.zeros((N,1),dtype=bool)

    X = np.vstack((class0,class1))
    Y = np.vstack((Y0,Y1)).ravel()

    rng = np.random.default_rng()
    arr = np.arange(2*N)
    rng.shuffle(arr)

    X = X[arr]
    Y = Y[arr]

    return X, Y, class0, class1


def nonlinear_dataset_13(N):
    mean0 = np.array([0, 0])
    cov0 = np.array([[3.0, 0.0],
                     [0.0, 0.15]])

    class0_raw = np.random.multivariate_normal(mean0, cov0, N)

    std_x = np.sqrt(cov0[0, 0])
    std_y = np.sqrt(cov0[1, 1])

    mask_x = np.abs(class0_raw[:, 0]) <= 3 * std_x
    mask_y = np.abs(class0_raw[:, 1]) <= 3 * std_y
    mask = mask_x & mask_y

    class0 = class0_raw[mask]

    theta = 2 * np.pi * np.random.rand(N)

    r_inner = 0.9
    r_outer = 1.1
    r = np.sqrt(np.random.uniform(r_inner**2, r_outer**2, N))

    a = 5.0
    b = 1.1

    x1 = a * r * np.cos(theta)
    y1 = b * r * np.sin(theta)

    class1 = np.column_stack((x1, y1))

    n_samples = min(len(class0), len(class1))

    X = np.vstack((class0[:n_samples], class1[:n_samples]))
    Y0 = np.zeros((n_samples, 1), dtype=bool)
    Y1 = np.ones((n_samples, 1), dtype=bool)
    Y = np.vstack((Y0, Y1)).ravel()

    rng = np.random.default_rng()
    idx = np.arange(2 * n_samples)
    rng.shuffle(idx)

    return X[idx], Y[idx], class0, class1
