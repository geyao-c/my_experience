n,k = map(int,input().split())
l = list(map(int,input().split()))
res = []
path = []
def func(l):
    l.sort()
    for i in range(1,len(l)):
        if l[i]%l[i-1]!=0:
            return False
    return True
def dfs(u):

    if len(path[:])==n-k:
        print("n - k: ", n - k)
        if func(path[:]):
            print(path[:])
            res.append(path[:])
        return
    for i in range(u,n):
        path.append(l[i])
        dfs(i+1)
        path.pop()

    return res

ans = dfs(0)
print(len(ans)%(10**9+7))
