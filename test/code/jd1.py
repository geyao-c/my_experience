

if __name__ == '__main__':
    n = int(input())
    a = [int(item) for item in input().strip().split(' ')]
    b = [int(item) for item in input().strip().split(' ')]

    a.sort(); b.sort()
    suma, sumb = 0, 0
    for i in range(n):
        if a[i] > b[i]:
            suma += a[i] - b[i]
        else:
            sumb += b[i] - a[i]

    if suma == sumb: print(suma)
    else: print(-1)