
if __name__ == '__main__':
    m, n = [int(item) for item in input().strip().split(' ')]
    a = [0] * (n + 1)
    while m > 0:
        l, r = [int(item) for item in input().strip().split(' ')]
        a[l] += 1; a[r + 1] -= 1
        m -= 1

    b, ans = 0, 0
    for i in range(0, n):
        b += a[i]
        if b == 0: ans += 1

    print(ans)


