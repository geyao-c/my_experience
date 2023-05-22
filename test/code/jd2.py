
if __name__ == '__main__':
    R, r = [int(item) for item in input().strip().split()]
    # pi = 3.141592654
    if R > r :
        ans = (R * R - (R - r) * (R - r)) / (R * R)
    else:
        ans = 1
    print(ans)