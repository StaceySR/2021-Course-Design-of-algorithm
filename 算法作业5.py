'''
1、最大子段和问题

问题描述：给定长度为n的整数序列，a[1…n], 求[1,n]某个子区间[i , j]，使得a[i]+…+a[j]和最大。

示例：输入：(-2,11,-4,13,-5,2)

           输出：最大子段和为20，所求子区间为[2,4]
'''

# 最简单的思路是：将序列中的每个元素和放在其后的元素的每个区间进行求和，比较这些和，将最大值输出。但是这个算法时间复杂度为n^2
# 第二个思路：套用分治法。将序列划分成左右两段，分别对左右两段求区间最大值。但是这个区间可能是在中间，也就是这个区间的左半边在左边那段，右半边在右边那段。因此还需要考虑跨序列的情况。从中间开始，不断往左和往右延伸，记录最大值。
# 第三个思路：动态规划。从序列的第一个元素开始，不断往下遍历，计算字段之和。有一个变量记录本次字段值和，一个变量一直记录和的最大值，一个变量记录字段左区间，一个变量记录字段右区间。在序列往后遍历的同时，如果遇到下一个元素加到先前字段和中，当前字段和为正，那么就加入，并且比较是否需要更新最大字段和；如果下一个元素加到先前字段和中，当前字段和为负，那么就将当前字段和清0，从下一个元素开始重新记录字段和，当然最大字段和不变


def get_max_sum(arr):
    now_sum = 0
    max_sum = 0  # max_sum初始值为0
    max_left = 1  # max_left初始值为1
    max_right = 1  # max_right初始值为1
    now_left = 1
    now_right = 1
    for i in range(len(arr)):
        now_sum += arr[i]
        now_right = i+1  # 不断更新当前字段的右区间
        if now_sum < 0:  # 加入当前元素之后，当前字段和小于0，那么就将当前字段和清0
            now_sum = 0
            now_left = i+1 + 1  # 更新当前字段的左区间，从下一个元素开始，所以要+1.因为要重启一个新字段
            now_right = i+1 + 1  # 更新当前字段的右区间，从下一个元素开始，所以要+1.因为要重启一个新字段
        if max_sum <= now_sum:  # 更新最大字段和，及其左右区间
            max_sum = now_sum
            max_left = now_left
            max_right = now_right
    if max_right == max_left and max_right > len(arr):  # 如果序列中所有元素都为负数，那么程序运行之后，max_left=max_right=len(arr)+1
        print('全员负数')
        return 0, 0, 0  # 如果序列中所有元素都为负数，那么就将max_sum, max_left, max_right，全赋为0
    return max_sum, max_left, max_right

#
# arr = [-1, -2]
# arr = [2]
# arr = [-2, 11, -4, 13, -5, 2]
# max_sum, max_left, max_right = get_max_sum(arr)
# gap = [max_left, max_right]
# print('最大子段和为：', max_sum, '，所求子区间为：', gap)
#

'''
2、拾捡硬币问题
问题描述：假如有n 个硬币排在一行，要求拾取其中的子序列，该序列的累加面值最大，但不能拾取相邻的两个硬币。
示例：输入5; 1; 2; 10; 6; 2，
           输出：Max=17 （5，10，2）
'''
# 看了视频之后的理解

# 设对于每个位置的硬币的最优解为OPT(i)
# 从最后一个硬币开始往前考虑，对于每个硬币来说，都可以有两种选择，捡或者不捡
#     捡       ：OPT(i-2) + arr[i]  (因为与之相邻的那个就不能捡了)
#     不捡     ：OPT(i-1)
#     那么对于这个位置的硬币的最优解OPT(i)=max{OPT(i-2) + arr[i],OPT(i-1)}
# 分析至此，递归式子就顺其自然地出来了，OPT(i)=max{OPT(i-2) + arr[i],OPT(i-1)}
# 然后找到递归地出口，OPT(0) = arr[0]; OPT(1) = max(arr[0],arr[1])

# # 递归方法
# def rec_OPT(arr, i):
#     if i == 0:
#         return arr[0]
#     elif i == 1:
#         return max(arr[0], arr[1])
#     else:
#         A = rec_OPT(arr, i-2) + arr[i]  # 捡第i枚硬币
#         B = rec_OPT(arr, i-1)  # 不见第i枚硬币
#         return max(A, B)


# arr = [5, 1, 2, 10, 6, 2]
# max = rec_OPT(arr, 5)
# print(max)

# 动态规划方法
import numpy as np
import copy


def dp_opt(arr, i):
    alist = [[0] for _ in range(6)]  # alist 用来存储每个位置地最有捡法（捡哪几个）
    opt = np.zeros(len(arr))  # 先创建一个最优解列表，记录对于每一个位置上及其之前地硬币中地拾取情况的最优解

    opt[0] = arr[0]  # 对于第0个位置上的硬币，累加最大，就是捡第0个
    alist[0] = [arr[0]]  # 第0个位置上的捡法，就是捡第0个

    opt[1] = max(arr[0], arr[1])  # 对于第1个位置上的硬币，累加最大，要看是捡第0个还是第1个
    if max(arr[0], arr[1]) == arr[0]:
        alist[1] = [arr[0]]  # 捡法：捡第0个
    else:
        alist[1] = [arr[1]]  # 捡法：捡第1个

    for i in range(2, len(arr)):
        A = opt[i-2] + arr[i]
        B = opt[i-1]
        if max(A, B) == A:
            alist[i] = copy.deepcopy(alist[i-2])  # 深拷贝，不然会出问题
            alist[i].append(arr[i])  # 捡法：捡第i个，依照前i-2个硬币地最优捡法，加上第i个硬币
        else:
            alist[i] = copy.deepcopy(alist[i-1])  # 捡法：不捡第i个，依照前i-1个硬币地最优捡法
        opt[i] = max(A, B)
    return opt[-1], alist[-1]  # 最终最优解


# arr = [5, 1, 2, 10, 6, 2]
# max, select = dp_opt(arr, 5)
# print('累加面值最大是：', max)
# print('选择捡：', select)



# # 题目有一个限制条件，不能拾取相邻的两个硬币，但又要保证序列的累加面值最大。因此在拾取每一个硬币之前，都要首先考虑该值要不要加。
# # 那么怎么考虑呢？就要将序列累加面值中有其前面相邻的元素的累加面值与序列累加面值中没有前面相邻的但有自己相加的累加面值相比；如果前者较大，那么就考虑该值先不要相加；如果后者较大，那么就考虑先加入该值。
# # 当然，上述情况都是动态的，即随着序列往后遍历，可能会产生变化。
#
# # list[::-1]    将列表list整个逆序输出
# # 对于第i个硬币，
# # 1）拾取第i个硬币，则table[i-2]+c[i]（ table[i-1]中包含了第i-1个硬币）
# # 2）不拾取第i个硬币，则table[i-1]
# # 取两者里边的最大值给了table[i]
#
# # 求table
# def coinamount(c):
#     table = [None] * (len(c) + 1)
#     table[0] = 0
#     table[1] = c[0]
#     for i in range(2, len(c) + 1):
#         table[i] = max(table[i - 2] + c[i - 1], table[i - 1])
#     return table
#
#
# # 回溯  找到究竟是拾取了哪些硬币
# def back(table, c):
#     select = []
#     lent = len(table) - 1
#     i = lent
#     while i >= 1:
#         if table[i] > table[i - 1]:  # 选最后一个
#             select.append(c[i - 1])
#             i -= 2  # 不取相邻的硬币
#         else:
#             i -= 1  # 考虑相邻的硬币
#     return select  # 最后一次拾取的硬币面值存放在select中的第一位
#
#
# if __name__ == "__main__":
#     c = [5, 1, 2, 10, 6, 2]
#     temp = coinamount(c)
#     select = back(temp, c)
#     print('拾取硬币最大的累加面值为：', temp[-1])  # 最后一位中存放着拾取硬币的累加面值
#     print('拾取了这些硬币：', select[::-1])  # 将拾取的硬币面值 按硬币一开始的顺序输出


'''
3、 矩阵连乘问题

问题描述：矩阵连乘问题是通过给矩阵连乘时加括号，使得总的计算量最小。

示例：输入：[[49, 29], [29, 32], [32, 77], [77, 13], [13, 59]]，

          输出：((A1(A2(A3A4)))A5)
'''


def matrix_chain(matrix, n, m, s):
    # 初始化记录矩阵
    for i in range(1, n+1):  # 方便对应
        m[i][i] = 0
        s[i][i] = 0
    for i in range(2, n+1):  # 求出A1A2，A2A3……两个矩阵的乘积次数，填入二维表
        m[i-1][i] = matrix[i-2][0] * matrix[i-2][1] * matrix[i-1][1]
        s[i-1][i] = i-1
    for i in range(n-2, 0, -1):  # 从右下角往上填！
        if i+2 <= n:
            for j in range(i+2, n+1):
                t = float('inf')  # 先设置t为无穷大，后面更小的t来覆盖
                for k in range(i, j):  # k为第几个间隔处分割
                    # 那么多分法中求最小值
                    if t > m[i][k] + m[k+1][j] + matrix[i-1][0]*matrix[k][0]*matrix[j-1][1]:
                        t = m[i][k] + m[k+1][j] + matrix[i-1][0]*matrix[k][0]*matrix[j-1][1]
                        s[i][j] = k
                m[i][j] = t


# 根据s矩阵中记录的分隔点k的位置，来输出
# Ai....Aj
def part(i, j, s):  # i代表第一个矩阵的下标，j代表最后一个矩阵的下标
    if i == j:
        print('A%s' % i, end='')
        return
    print('(', end='')
    # 根据分隔点的位置，将所有矩阵分成左右两部分，分别对两部分进行递归输出分割点
    part(i, s[i][j], s)  # 左半边递归
    part(s[i][j]+1, j, s)  # 右半边递归
    print(')', end='')

#
# matrix = [[49, 29], [29, 32], [32, 77], [77, 13], [13, 59]]
# n = len(matrix)  # matrix矩阵个数
# m = []  # 记录矩阵，如m[i][j]记录为Ai*……*Aj矩阵的最少乘积次数
# s = []
# for i in range(0, n+1):  # 初始化记录矩阵，为了方便理解，第0行和第0列是不用的
#     m.append([])
#     s.append([])
#     for j in range(0, n+1):
#         m[i].append(0)
#         s[i].append(0)
#
#
# matrix_chain(matrix, n, m, s)
# print('m:')
# for i in range(1, n+1):
#     print(m[i][1:])
#
# print('分隔点k所处位置的矩阵s:')
# for i in range(1, n+1):
#     print(s[i][1:])
#
# # 输出分隔情况
# part(1, n, s)







'''
4、最短公共超序列问题

问题描述：给出两个字符串str1和str2，返回同时以str1和str2作为子序列的最短字符串。
如果答案不止一个，则可以返回满足条件的任意一个答案。如果从字符串 T 中删除一些字符
（也可能不删除，并且选出的这些字符可以位于T中的任意位置），可以得到字符串 S，
那么S就是T的子序列。设1<=str1.length, str2.length<=1000，str1和str2都由小写英文字母组成。

示例：输入：str1 = "abac", str2 = "cab"      

           输出："cabac"

          解释：str1 = "abac" 是 "cabac" 的一个子串，因为可以删去 "cabac" 的第一个 "c"得到 "abac"。
           str2 = "cab" 是 "cabac" 的一个子串，因为可以删去 "cabac" 末尾的 "ac" 得到 "cab"。
           最终给出的答案是满足上述属性的最短字符串。

'''


# 先求最短公共子序列
def Lcs(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    c = [[0 for i in range(len2+1)]for j in range(len1+1)]  # 找最长公共子序列最大长度的二维表
    flag = [[0 for i in range(len2+1)]for j in range(len1+1)]  # 记录路径的二维表，之后回溯找公共子序列要用到
    for i in range(len1):
        for j in range(len2):
            if str1[i] == str2[j]:
                c[i + 1][j + 1] = c[i][j] + 1
                flag[i+1][j+1] = 'ok'  # 选择的是左上角格子的内容
            elif c[i+1][j] > c[i][j+1]:  # 左边格子的数大于上边格子的数，选择较大的那个
                c[i+1][j+1] = c[i+1][j]
                flag[i+1][j+1] = 'left'  # 左边格子较大，选择左边
            else:  # 上边格子大于等于左边格子时，都选择上边，只考虑一种情况
                c[i+1][j+1] = c[i][j+1]
                flag[i+1][j+1] = 'up'
    return c, flag


global common_str  # 最长公共子序列
common_str = ''


# 回溯函数，回溯找最长公共子序列，把它记录在common_str中
def printLcs(flag, a, i, j):  # 从二维表右下角开始回溯路径
    global common_str
    if i == 0 or j == 0:
        return
    if flag[i][j] == 'ok':  # 这个位置的字母是公共子序列中的一个
        printLcs(flag, a, i-1, j-1)
        common_str += a[i-1]
    elif flag[i][j] == 'left':
        printLcs(flag, a, i, j-1)  # 选择左边的区域递归
    else:
        printLcs(flag, a, i-1, j)  # 选择上边的区域递归


# 在公共子序列的基础上再找最短公共超序列
def find_shortest_common_supersequence(common_str, str1, str2):  # 在最长公共子序列的基础上拼接上去其他字母，形成最短公共超序列
    i = 0
    j = 0
    res = ''
    for c in common_str:  # 分别取出最长公共子序列中的每一个字母，与str1和str2中的每个字母进行比较把不相等的字母都加入到最短公共超序列中
        while str1[i] != c:
            res += str1[i]
            i += 1
        while str2[j] != c:
            res += str2[j]
            j += 1
        res += c
        i += 1
        j += 1
    return res + str1[i:] + str2[j:]  # 最长公共子序列已经遍历完了，接下来把str1和str2中剩下未遍历的字母都加入到最短公共超序列中
#
#
# str1 = 'abac'
# str2 = 'cab'
# c, flag = Lcs(str1, str2)
# # print('最长公共子序列的长度：', c[-1][-1])
# printLcs(flag, str1, len(str1), len(str2))
# # print('最长公共子序列：', common_str)
# print('最短公共超序列：', find_shortest_common_supersequence(common_str, str1, str2))
#
#



'''
5、对于一个n=5的关键字集合，搜索概率如下表，请构造其最优二分搜索树。


'''


def optimal_bst(p, q, n):
    e = [[0 for j in range(n + 1)] for i in range(n + 2)]  # 记录最优二叉树的搜索代价
    w = [[0 for j in range(n + 1)] for i in range(n + 2)]  # w[i,j]为所有概率之和p1+p2…+pn + q0+q1+…+qn
    root = [[0 for j in range(n + 1)] for i in range(n + 1)]  # 记录最优二叉搜索树的根的位置

    # 初始化
    for i in range(n + 2):
        e[i][i - 1] = q[i - 1]
        w[i][i - 1] = q[i - 1]

    for l in range(1, n + 1):  # l为i到j的跨度  e[i][j]
        for i in range(1, n - l + 2):
            j = i + l - 1
            e[i][j] = float("inf")
            w[i][j] = w[i][j - 1] + p[j] + q[j]
            for r in range(i, j + 1):  # 根结点从第一个到最后一个都计算一遍
                t = e[i][r - 1] + e[r + 1][j] + w[i][j]  # 递推公式：左子树的最优平均比较次数 + 右子树的最有平均比较次数 + 两者上升一层后需要增加的次数
                if t < e[i][j]:
                    e[i][j] = t
                    root[i][j] = r
    return e, root

#
# if __name__ == "__main__":
#     p = [0, 0.15, 0.1, 0.05, 0.1, 0.2]
#     q = [0.05, 0.1, 0.05, 0.05, 0.05, 0.1]
#     e, root = optimal_bst(p, q, 5)
#     print('最优二叉树的搜索代价矩阵:')
#     for i in range(5 + 2):
#         for j in range(5 + 1):
#             print(e[i][j], " ", end='')
#         print()
#     print('最优二叉搜索树的根的位置矩阵:')
#     for i in range(5 + 1):
#         for j in range(5 + 1):
#             print(root[i][j], " ", end='')
#         print()

'''
6、买卖股票的最佳时机

问题描述：给定一个数组，它的第i个元素是一支给定的股票在第i天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成k笔交易。注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
示例1：输入: [2,4,1], k = 2
              输出: 2
解释: 在第 1 天 (股票价格=2)的时候买入，在第2天 (股票价格=4)的时候卖出，这笔交易所能获得利润=4-2=2 。
示例2：输入: [3,2,6,5,0,3], k = 2

             输出: 7
解释: 在第 2 天 (股票价格=2) 的时候买入，在第3天 (股票价格=6)的时候卖出, 这笔交易所能获得利润= 6-2=4。
随后，在第5天(股票价格=0)的时候买入，在第6天(股票价格=3)的时候卖出, 这笔交易所能获得利润=3-0=3
'''


def max_profit(price, K):
    n = len(price)
    if n == 0:
        return 0
    # 定义一个三维数组
    # i表示第i天，k表示几次交易，j表示是否持股(j=0不持股，j=1持股）
    # 注意，买、卖这整个流程才算一次交易，从买入算一次交易
    profit = [[[0 for _ in range(2)] for _ in range(K+1)] for _ in range(n)]

    for i in range(0, n):
        for k in range(K, 0, -1):
            if i == 0:  # 第一天  初始化
                profit[i][k][0] = 0  # 第一天操作k次，但不持股，所以初始值为0
                profit[i][k][1] = -price[i]  # 第一天操作k次，持股票，所以初始值为-price[0]
                continue
            # 不持股
            profit[i][k][0] = max(profit[i-1][k][0], profit[i-1][k][1]+price[i])  # 不动 | 卖出，赚钱
            # 持股
            profit[i][k][1] = max(profit[i-1][k][1], profit[i-1][k-1][0]-price[i])  # 不动 | 买入，付钱

    return profit[n-1][K][0]

#
# prices = [3, 2, 6, 5, 0, 3]
# maxP = max_profit(prices, 2)
# print(maxP)


#
# def maxProfit(prices, K):
#     if not prices: return 0
#     n = len(prices)
#     dp = [[[0 for i in range(2)] for i in range(K+1)] for i in range(n)]
#     for k in range(K+1):
#         dp[0][k][1] = -float('inf')
#         dp[-1][k][1] = -float('inf')
#     for i in range(n):
#         for k in range(1, K + 1):
#             dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
#             dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
#
#     return dp[n - 1][K][0]
#
#
# prices = [2, 4, 1]
# maxP = maxProfit(prices, 2)
# print(maxP)


'''
7、天平秤金条问题

问题描述：有30根金条，其中一根比其它的要重，请问用一个天平至少秤几次可以将这个重的金条找出来。

示例1：输入：[10, 10, 10, 10, 10, 11]

             输出：The fake coin is coin 6 in the original list             Number of weighings: 2

示例2：输入：[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 10, 10] 

              输出：The fake coin is coin 27 in the original list             Number of weighings: 3

'''

# 将所有金条按顺序分成三份，前两份数量相等，最后一份可能数量和前两份相等或者比他们多一份或者两份
# 首先比较前面两份数量相等的金条，将其分别放到天平的两边，这算称一次，如果相等，那么较重的那根金条一定在第三份中；如果不相等，那么就一定在前面两份较重的那份中
# 可以以这样的方式不断缩小金条数量，递归进行。
# 递归的出口就是只剩0根金条时，那么就不用再继续比较；只剩1根金条时，那么就不用再继续比较，直接找到了；只剩2根时，那么就只要再比较一次就可以找到


def find_one(gold, time):  # time为天平称几次
    n = len(gold)
    if n == 0:
        return time
    if n == 1:  # 只有两个数时，不用再比较
        return time
    if n == 2:  # 只有两个数时，只要比较一次即可
        return time+1
    apart = n//3  # 将金条按顺序分成三份
    part1 = 0
    part2 = 0

    for i in range(apart):  # 将第一份和第二份金条进行重量比较
        part1 += gold[i]
        part2 += gold[i+apart]
    time += 1  # 比较完一次，就是天平称次数加1
    if part1 == part2:  # 如果第一份和第二份金条重量相等
        return find_one(gold[2*apart:], time)  # 对第三份金条进行递归
    elif part1 > part2:  # 如果第一份金条较重
        return find_one(gold[0:apart-1], time)  # 对第一份金条进行递归
    else:  # 如果第二份金条较重
        return find_one(gold[apart:2*apart-1], time)  # 对第二份金条进行递归
#
#
# gold = [10, 10, 10, 10, 10, 11]
# print('The fake coin is coin %d in the original list       '
#       'Number of weighings:' % len(gold),  find_one(gold, 0))  # 要一开始要设置比较次数为0
#
#


'''
8、动态规划解最短路径问题

问题描述：从某顶点出发，沿图的边到达另一顶点所经过的路径中，求各边上权值之和最小的一条路径——最短路径。 

示例：输入如下图（图的输入形式自行确定）：

输出：从A到G的最短路径长度为： 18      经过的结点为： ['B1', 'C2', 'D1', 'E2', 'F2']

'''

G = {1: {1: 0, 2: 1, 3: 12},
     2: {2: 0, 3: 9, 4: 3},
     3: {3: 0, 5: 5},
     4: {3: 4, 4: 0, 5: 13, 6: 15},
     5: {5: 0, 6: 4},
     6: {6: 0}}


def Dijkstra(G, v0, INF=999):  # 999表示该结点到起始点的距离还没有一个确切的值
    dis = dict((i, INF) for i in G.keys())  # 初始化一个距离表，这个表记录着起始结点v0到图中各个点的距离
    current_node = v0  # 一开始，当前点设置为起始点v0
    dis[v0] = 0  # 初始点到自己的距离自然为0
    visited = []  # 记录已经遍历过的结点
    ###
    path = dict((i, []) for i in G.keys())  # 初始化一个路径表，这个表记录着起始结点到途中各个点的最短路径
    path[v0] = str(v0)  # 初始点到自己的路径自然为自己
    ###

    while len(G) > len(visited):  # 图的结点还没被遍历完时执行循环以继续遍历
        visited.append(current_node)  # 当前点正在被遍历，所以把当前点放入visited表中
        for k in G[current_node]:  # 遍历当前点的所有相邻点
            if dis[current_node] + G[current_node][k] < dis[k]:  # 如果（起始点到当前点的相邻点k的距离）大于（起始点到当前点的距离+当前点到相邻点k的距离）
                dis[k] = dis[current_node] + G[current_node][k]  # 则（起始点到当前点的相邻点k的距离）更新为（起始点到当前点的距离+当前点到相邻点k的距离）
                seq = (path[current_node], str(k))
                sym = '-'
                path[k] = sym.join(seq)  # 起始点到（当前点的相邻点k）的最短路径，以'-'来连接seq中的两个字符串
        # 接着来选下一个当前点(current_node)
        # 从剩下未遍历的点中，选取与当前点的距离最小的那个点作为下一个当前点
        new = INF
        for node in dis.keys():
            if node in visited: continue
            if dis[node] < new:
                new = dis[node]
                current_node = node
    return dis, path


dis, path = Dijkstra(G, v0=1)
print(dis)
print(path)
print('从1到6的最短路径是：%s' % dis[6])
print('经过的结点为：', path[6])








#
# import heapq
#
# graph = {
#     "A":{"B1":5,"B2":3},
#     "B1":{"A":5,"C1":1,"C2":3,"C3":6},
#     "B2":{"A":3,"C2":8,"C3":7,"C4":6},
#     "C1":{"B1":1,"D1":6,"D2":8},
#     "C2":{"B1":3,"B2":8,"D1":3,"D2":5},
#     "C3":{"B1":6,"B2":7,"D2":3,"D3":3},
#     "C4":{"B2":6,"D2":8,"D3":4},
#     "D1":{"C1":6,"C2":3,"E1":2,"E2":2},
#     "D2":{"C1":8,"C2":5,"C3":3,"C4":8,"E2":1,"E3":2},
#     "D3":{"C3":3,"C4":4,"E2":3,"E3":3},
#     "E1":{"D1":2,"F1":3,"F2":5},
#     "E2":{"D1":2,"D2":1,"D3":3,"F1":5,"F2":2},
#     "E3":{"D3":3,"D2":2,"F1":6,"F2":6},
#     "F1":{"E1":3,"E2":5,"E3":6,"G":4},
#     "F2":{"E1":5,"E2":2,"E3":6,"G":3},
#     "G":{"F1":4,"F2":3}
# }
#
#
#
# class Dijkstra:
#     def init_distance(self, graph, start):
#         distance = {start: 0}
#         for key in graph.keys():
#             if key != start:
#                 distance[key] = float('inf')
#         return distance
#
#     def dijkstra(self, graph, start):
#         if not graph or not start:
#             return None
#
#         distance = self.init_distance(graph, start)
#         pqueue = []
#         heapq.heappush(pqueue, (0, start))
#         seen = set()
#         parent = {start: None}
#
#         while pqueue:
#             cur_distance, cur_node = heapq.heappop(pqueue)
#             seen.add(cur_node)
#             nodes = graph[cur_node]
#
#             for node, dist in nodes.items():
#                 if node in seen:
#                     continue
#                 elif distance[node] > cur_distance + dist:
#                     heapq.heappush(pqueue, (dist + cur_distance, node))
#                     parent[node] = cur_node
#                     distance[node] = cur_distance + dist
#         return distance, parent
#
#
# if __name__ == '__main__':
#     s = Dijkstra()
#     res, parent = s.dijkstra(graph, "A")
#     print(res)
#     print(parent)
#

