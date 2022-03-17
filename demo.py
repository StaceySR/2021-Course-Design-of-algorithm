# tsp问题


import numpy as np


class Solution:
    def __init__(self, X, start_node):
        self.X = X  # 距离矩阵(需要进行动态规划)
        self.start_node = start_node  # 开始的节点
        self.array = [[0] * (2 ** (len(self.X) - 1)) for i in range(len(self.X))]  # 记录处于x节点，未经历M个节点时，矩阵储存x的下一步是M中哪个节点

    def transfer(self, sets):
        su = 0
        for s in sets:
            su = su + 2 ** (s - 1)  # 二进制转换
        return su

    # tsp总接口
    def tsp(self):
        s = self.start_node  #起始节点
        num = len(self.X)    #总结点个数
        cities = list(range(num))  # 形成节点的集合
        # past_sets = [s] #已遍历节点集合
        cities.pop(cities.index(s))  # 构建未经历遍历的节点的集合，cities中剩下的是还没有遍历过的节点
        node = s  # 当前遍历的初始节点
        return self.solve(node, cities)  # 求解函数

    def solve(self, node, future_sets):
        # 迭代终止条件，表示没有了未遍历节点，直接连接当前节点和起点即可
        if len(future_sets) == 0:
            return self.X[node][self.start_node]
        d = 99999
        # node如果经过future_sets中节点，最后回到原点的距离
        distance = []
        # 遍历未经历的节点
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            copy = future_sets[:]
            copy.pop(i)  # 删除第i个节点，认为已经完成对其的访问
            distance.append(self.X[node][s_i] + self.solve(s_i, copy))
        # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 未遍历节点集合
        c = self.transfer(future_sets)
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.array[node][c] = next_one
        return d


def matrix_creator(i): #i代表生成矩阵的大小
    #  随机对称矩阵生成函数
    matrix = np.random.randint(0, 70, (i, i))
    matrix = np.triu(matrix)
    matrix += matrix.T - np.diag(matrix.diagonal())
    for i in range(len(matrix)):
        matrix[i][i] = -1
    return matrix


def dataload(client,D):
    '''
    生成顾客需求矩阵
    '''

    delete = []
    # matrix_creator(7)#有7个快递柜
    for i in range(len(D)):
        if i not in client:
           delete.append(i)


    D = np.delete(D,delete, axis = 1)  # 删除第i列
    D = np.delete(D, delete, axis=0)  # 删除第i行


    return D





if __name__ == '__main__':
    #
    # D=[
    #     [-1,15,23,30,15,45,24],
    #     [15,-1,10,22,3,46,35],
    #     [23,10,-1,40,51,60,62],
    #     [30,22,40,-1,32,15,19],
    #     [15,3,51,32,-1,12,38],
    #     [45,46,60,15,12,-1,52],
    #     [24,35,62,19,38,52,-1]
    # ]

    D = matrix_creator(10)
    print("快递投放点邻接矩阵为：")
    print(D)
    client = []
    print("请输入需要途径的目的地址（首先输入原点）： ")
    while True:
        res = input()
        if res=='#':
            break

        else:
            if res.isdigit():
                client.append(int(res))

    client.sort()
    D= dataload(client,D)
    print("按照目的地址重新生成的用户需求邻接矩阵： ")
    print(D)


    S = Solution(D, 0)
    print("快递员经过的最短距离为： ",S.tsp())
    # 开始回溯
    M = S.array
    lists = list(range(len(S.X)))
    start = S.start_node
    print()
    print("投递路径为： ")
    while len(lists) > 0:
        lists.pop(lists.index(start))
        m = S.transfer(lists)
        next_node = S.array[start][m]
        if start==0:
            print("物流点", "--->", str(client[next_node])+"号地点")
        elif next_node==0:
            print(str(client[start])+"号地点", "--->", "物流点")
        else:
            print(str(client[start])+"号地点", "--->", str(client[next_node])+"号地点")
        start = next_node