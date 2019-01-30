def memory_dense(dims):
    ni, lr, lc = dims
    Ni = ni * lr * lc
    return Ni**2 + 2 * Ni


def compute_dense(dims):
    ni, lr, lc = dims
    Ni = ni * lr * lc
    return Ni**2

if __name__ == '__main__':
    inputs = [
        (1, 28, 28),
        (3, 32, 32),
        (3, 224, 224),
        (3, 512, 1024),
        (3, 1024, 2048)
    ]

    for inp in inputs:
        print('Memory: %i' % memory_dense(inp) )
        print('Compute: %i' % compute_dense(inp) )
        print('Total: %i' % (compute_dense(inp) + memory_dense(inp)) )

