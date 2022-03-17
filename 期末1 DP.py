'''

n : 总共有几个城市的航班记录
flights : 一个列表，记录各航班信息，如 list = ['121', '221', '321', '421']
                                    for a, b, c in list:
                                        print(a, b, c)
src ： 记录始发地
dst ： 记录目的地
K ： 记录最多经过几次转机，在这个转机次数内，取最便宜的
      /*
         * 题目概述：n 个城市 m 个航班 k 次中转从一座城市到达另一座城市,最便宜的价格是多少
         *
         * 思路：
         *  1. 定义状态:dp[i][n] 表示第K 次中转时,目前可达所有城市的最小价格
         *  2. 结果状态:dp[i][dst]
         *  3. 状态转移:dp[i][city] = min(dp[i][cityOthers]+prices) (cityOthers 是到达其他城市以后,再飞到 city 城市的航班)
         *
         * 关键点：
         *
         * 时间复杂度：O(flights) flights 即航班的数量
         * 空间复杂度：O(n)
         */


'''

#
# def findCheapestPrice(n, flights, src, dst, K):
#     # dp[k][d] 记录经过k次转机后到达d城市的最小花费
#     dp = [[float('inf')]*n for i in range(K+1)]
#     # 初始化0次转机直达的花费
#     for s, d, p in flights:
#         if s == src:
#             dp[0][d] = p
#     # 1次及以上转机的花费
#     for k in range(1, K+1):
#         for s, d, p in flights:  # dp[k-1][s]+p的意思是，上一次先飞到了s城市，现在要从s城市飞往d城市
#             # 不需要考虑太多，只需要记住，左边数轴是代表第几次转机，右边上面横轴代表是到哪个城市（可能到达d城市可以有不同的转机次数，那么转机路线也不同，因此不用管那么细）
#             # 动态规划方程，状态转移，第k次转机到达d城市的最小花费 = min (第k-1次转机到达d城市的最小花费,第k-1次转机到达s城市的最小花费 +s到达d的花费p,第k次转机到达d城市的最小花费)
#             dp[k][d] = min(dp[k-1][d], dp[k-1][s]+p, dp[k][d])
#     print(dp)
#     if dp[K][dst] != float('inf'):
#         return dp[K][dst]  # 返回最后一行的目的地址对应的格子中的数据
#     else:
#         return -1
#     # return dp[-1][dst] if dp[-1][dst] != float('inf') else -1


'''
dp[k][dst]表示从src经过k次中转到达dst的最便宜价格，注意src是固定的，所以不需要维护，只需要维护dp[k][dst]
注意 fight = [src,dst,price],src,dst,price = fight[0],fight[1],fight[2]
递推方程：dp[k][dst] = min(dp[k][dst],dp[k-1][src]]+price)
初始条件一：整个二维表格初始化为 float('inf'),表示从src经过k次中转无法到达dst
初始条件二：遍历flight，找到出发点为src的航班，标记dp[0][fight[1]]==fight[2],即更新src可直达航班
初始条件三：dp[k][src]=0,因为从出发点到出发点不管经过几次中转，都应该认为价格是0
'''


def findCheapestPrice(n, flights, src, dst, K):
    dp = [[float('inf')] * n for _ in range(K + 1)]

    for flight in flights:
        if flight[0] == src:
            dp[0][flight[1]] = flight[2]

    for k in range(0, K + 1):
        dp[k][src] = 0

    for k in range(1, K + 1):
        for flight in flights:
            if dp[k - 1][flight[0]] != float('inf'):
                dp[k][flight[1]] = min(dp[k][flight[1]], dp[k - 1][flight[0]] + flight[2])
    return dp[K][dst] if dp[K][dst] != float('inf') else -1

#
# def finCheapestFlyLine(n, src, dst, flights, dp, K):
#     for k in range(K, -1):
#         if dp[k][dst] > dp[K][dst]:
#             index = k + 1
#             break
#     for k in range(0, K+1):
#         for i in range(0, n):
#             if dp[k][i] <= dp[K][dst]:







'''
各城市代号：
    Beijing : 0
    Shanghai : 1
    Hangzhou : 2
    Guangzhou : 3
    Haikou : 4
'''


#       始发地、目的地、价格
flights = [[0, 1, 300, 791000, 791220],
           [0, 2, 250, 791100, 791310],
           [0, 3, 780, 791320, 791500],
           [0, 4, 990, 792300, 7100140],
           [1, 0, 320, 791500, 791640],
           [1, 2, 170, 791200, 791320],
           [1, 3, 430, 791000, 791220],
           [1, 4, 560, 791230, 791500],
           [2, 0, 220, 790800, 791030],
           [2, 1, 120, 792350, 7100200],
           [2, 3, 480, 791245, 791500],
           [2, 4, 350, 790855, 791100],
           [3, 0, 275, 790145, 790340],
           [3, 1, 380, 790630, 790900],
           [3, 2, 310, 791200, 791430],
           [3, 4, 370, 791140, 791330],
           [4, 0, 904, 791935, 792100],
           [4, 1, 500, 791850, 792000],
           [4, 2, 440, 791420, 791700],
           [4, 3, 320, 791530, 791700]
           ]


print(findCheapestPrice(5, flights, 0, 4, 3))


# #            始发地、    目的地、   价格
# flights = [['Beijing', 'Shaihai', 300],
#            ['Hangzhou', 'Haikou', 350],
#            ['Beijing','Haikou', 680],
#            ['Haikou', 'Shanghai', 300],
#            ['Haikou', 'Beijing', 890],
#            ['Shanghai', 'Beijing', 210],
#            ['Hangzhou', 'Guangzhou', 180],
#            ['Guangzhou', 'Haikou', 160]
#            ]

