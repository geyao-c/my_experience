from collections import deque

class tree():
    def __init__(self, val):
        self.data = val
        self.lchild = None
        self.rchild = None

def array_to_tree(tlist):
    if tlist[0] == None:
        return None
    root = tree(tlist[0])
    Nodes = [root]
    j = 1
    for node in Nodes:
        if node != None:
            node.lchild = (tree(tlist[j]) if tlist[j] != None else None)
            Nodes.append(node.lchild)
            j += 1
            if j == len(tlist):
                print('j: ', j)
                return root
            node.rchild = (tree(tlist[j]) if tlist[j] != None else None)
            j += 1
            Nodes.append(node.rchild)
            if j == len(tlist):
                # print('jj: ', j)
                return root
    return root

def tree2array(root):
    alist = []
    queue = deque()
    queue.append(root)
    while len(queue) > 0:
        p = queue.popleft()
        if p == None:
            alist.append(-1)
            continue
        alist.append(p.data)
        queue.append(p.lchild)
        queue.append(p.rchild)
    return alist, id(root)

ans = []

def over_traver(root, rlist, rid):
    global ans
    if root is None:
        return None
    r2list, r2id = tree2array(root)
    if r2id != rid:
        len1 = len(rlist)
        len2 = len(r2list)
        if (len1 == len2):
            ok = True
            for i in range(len1):
                if rlist[i] != r2list[i]:
                    ok = False
                    break
            if ok:
                if len(rlist) > len(ans):
                    ans = rlist
    over_traver(root.lchild, rlist, rid)
    over_traver(root.rchild, rlist, rid)

def traver(root):
    global groot
    if root is None:
        return None
    rlist, rid = tree2array(root)
    over_traver(groot, rlist, rid)
    traver(root.lchild)
    traver(root.rchild)

array = input().strip('[]').split(',')
new_array = []
for item in array:
    if item != 'null':
        new_array.append(int(item))
    else:
        new_array.append(None)
print('new array: ', new_array)
groot = array_to_tree(new_array)
nnew = tree2array(groot)
print(nnew)
traver(groot)
print('ans: ', ans)

# [1,2,3,1,null,2,null,null,null,1,null,null,null]
# [2,1,null]