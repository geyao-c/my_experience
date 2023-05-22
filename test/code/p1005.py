import numpy as np

def sol(x):
    dp = [[0 for _ in range(m)] for _ in range(m)]
    # dp = np.zeros((m, m), dtype=int)
    for i in range(0, m): dp[i][i] = x[i] * (1 << m)
    for d in range(2, m + 1):
        for l in range(0, m):
            r = l + d - 1
            if r >= m: break
            lval = dp[l][r - 1] + x[r] * (1 << (m - d + 1))
            rval = dp[l + 1][r] + x[l] * (1 << (m - d + 1))
            dp[l][r] = max(lval, rval)
    return dp[0][m - 1]

if __name__ == '__main__':
    n, m = [int(item) for item in input().strip().split(' ')]
    ve = []
    sum = 0
    for i in range(n):
        tlist = [int(item) for item in input().strip().split(' ')]
        ve.append(tlist)
        sum += sol(ve[i])
    print(sum)