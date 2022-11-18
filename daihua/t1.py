
def solution(inputlogs):
    import functools
    def dcmp(x, y):
        if x[2] > y[2]: return 1
        else: return -1;

    goods, ans = [], []
    for log in inputlogs:
        if log[0] == 'supply':
            goods.append([log[1], int(log[2]), int(log[3])])
        elif log[0] == 'return':
            goods.append([log[1], int(log[2]), int(log[4])])
        elif log[0] == 'sell':
            item, num, cost = log[1], int(log[2]), 0
            # 自定义排序
            goods.sort(key=functools.cmp_to_key(dcmp))
            for good in goods:
                if num == 0:
                    break
                if good[0] == item:
                    tmp = min(good[1], num)
                    cost += tmp * good[2]
                    num -= tmp; good[1] -= tmp
            ans.append(cost)
            tmpgoods = []
            for good in goods:
                if good[1] != 0: tmpgoods.append(good)
            goods = tmpgoods
    return ans

if __name__ == '__main__':
    logs = [
        ['supply', 'item1', '2', '100'],
        ['supply', 'item2', '3', '60'],
        ['sell', 'item1', '1'],
        ['sell', 'item1', '1'],
        ['sell', 'item2', '2'],
        ['return', 'item2', '1', '60', '40'],
        ['sell', 'item2', '1'],
        ['sell', 'item2', '1'],
    ]
    ans = solution(logs)
    print(ans)