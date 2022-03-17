'''
1、从数组 seq 中找出和为 s 的数值组合，有多少种可能，并打印这些组合。

（1）问题描述：求一个算法：N个数，用其中M个任意组合相加等于一个已知数X。得出这M个数是哪些数。

（2）问题举例：seq = [1, 2, 3, 4, 5, 6, 7, 8, 9] s = 14 则全部可能的数字组合有：
        5+9, 6+8 1+4+9, 1+5+8, 1+6+7, 2+3+9, 2+4+8, 2+5+7, 3+4+7, 3+5+6 1+2+5+6, 1+3+4+6, 1+2+4+7, 1+2+3+8, 2+3+4+5 共计15种。
'''

n = 0  # 全局变量 来计数总共有多少种可能组合
def find2(seq, s, tmp=''):
    global n
    if len(seq) == 0:  # 序列里面没有数，就结束程序
        return
    if seq[0] == s:  # 找到一种，则
        print(tmp + str(seq[0]))  # 打印
        n += 1
    find2(seq[1:], s, tmp)  # 尾递归 ---不含 seq[0] 的情况
    find2(seq[1:], s - seq[0], str(seq[0]) + '+' + tmp)  # 尾递归 ---含 seq[0] 的情况


seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
s = 14
find2(seq, s)
print('总共有', n, '组')



'''
2、编程实现统计逆序问题

（1）问题描述：豆瓣是一家图书、电影和音乐唱片的评价与推荐网站。这类推荐类网站会根据你对系列书籍的评价，
从它的读者数据库中找出与你的评价非常类似的读者推荐给你，从而帮助你找到品味相近的朋友。
假设你对五本书进行了评价，这五本书你的打分从低到高依次是[1，2，3，4，5]。另外，
读者A的对这五本书的打分是[2，4，1，3，5]，而读者B的打分是[3，4，1，5，2]。那么，
应该把读者A还是读者B推荐给你呢? 豆瓣也许会把读者A推荐给你，因为相比较于读者B，
读者A与你的口味更为相投。那怎么来量化推荐的准则呢?

这可以通过计算一个称为逆序量的来度量相似度。对于输入序列，如果元素的索引 i＜j，且ai＞aj，那么元素ai和aj是一对逆序。
打分[1，2，3，4，5]的逆序对数为0，读者A打分[2，4，1，3，5]存在3对逆序，分别是[2，1],[4，1]和[4，3]。
读者B打分[3，4，1，5，2]的逆序数为5对，分别是[3，1]，[3，2]，[4，1],[4，2]和[5，2]。
因此，如果用逆序数来度量推荐准则，那么读者A相比较于读者B与你有更为接近的品位。
本问题就是计算给定序列的逆序数。

（2）算法设计：一个简单直接的算法就是对于每一个元素，计算该元素右边有几个元素比它小。
例如，对于输入序列[2，4，1，3，5]，元素2的右边共有1个元素叫比它小，元素4的右边 共有2个元素[1，3]比它小。因此，以上序列共有3对逆序。
'''
# 将一个序列分成两个序列 ：A[0..n/2]  A[n/2+1..n-1]
# 对分成两半的序列分别递归调用计算一半的逆序数
# 之后再比较位于左边和右边的数
#     如何比较呢？ 假设左半边和右半边已经排好序，那么就会大大减少比较次数
#     2531 7468
#     1235 4678    对两边排好序之后，左边的第一个数1和右边的第一个数4进行比较，1<4 ，那么1就不需要再和4后面的数进行比较了，因为4后面的数都比4大
#     并且对于求逆序数的对数而言，5678 4567 ，左边第一个数5比右边第一个数4大，那么逆序数就要增加len（left）=4，之后从左边第二个数刚开始与4比较，6》4，那么逆序数又增加len（left）（为了这样实现循环，那么就要把5从左边的序列中删除，以此类推）

def left_right_part(arr):
    if len(arr) == 1:
        return arr, 0  # 如果序列的长度为1，那么逆序数一定不存在
    mid = len(arr)//2  # 找到序列中间位置，将序列划分成左、右两个部分
    left, count_left = left_right_part(arr[:mid])  # 对于左半边的序列进行递归
    right, count_right = left_right_part(arr[mid:])  # 对于右半边的序列进行递归
    sort_arr, count_both = both_in_two_parts(left, right)  # 对于分散在左右两边的逆序对进行统计
    return sort_arr, count_left + count_right + count_both  # result是将序列排好序后的列表；左边序列的逆序对数+右边序列的逆序对数+分散在两边的逆序对数=总共的逆序对数


def both_in_two_parts(left, right):  # 对于逆序对元素分散在左右两边的情况进行逆序对数统计
    result = []
    count = 0
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            result.append(left.pop(0))  # 每次都把较小的加入到result，最后形成的就是一个从小到大的有序序列
            # left.pop(0)  # 因为已经将左边序列的第一个数加入到result列表中。并且为了之后else中求分散在两边的逆序数对数可以直接用len（left），所以这里直接将左边比右边小的数删除
        else:
            count += len(left)
            result.append(right.pop(0))  # 右边较小，那么就把右边的数加到result最后
            # count += len(left)  # 左边序列第一个元素比右边序列第一个元素大，因为左边已经是从小到大排好序的，那么就说明左边的所有元素都比右边第一个元素大，那么逆序对数count直接增加len（left）
            # right.pop(0)  # 将已经比较国地右边序列的第一个元素删除，以便比较后续的元素

    if len(left) != 0:  # 退出循环，最后将剩下的有序序列加入result中，整个result呈现出从小到大排列的有序状态
        result += left
    else:
        result += right
    return result, count


# arr = [3, 8, 5, 7, 2, 4, 9, 1]
# result, num = left_right_part(arr)
# print('排好序的序列为：', result)
# print('逆序对数为：', num)



'''
3、编程实现最大间隙问题

（1）问题描述：给定n个实数x1,x2, ,xn,求这n个数在实轴上相邻2个数之间的最大差值。要求设计出线性时间算法O(n)。

（2）问题举例：输入数据：5
                       2.3，3.1，7.5，1.5，6.3
            输出数据：3.2
'''
# 首先将最小值和最大值找出来，之后在最小值和最大值之间划分等区间  size = (max_val - min_val) / (n - 1)
# 首先每个区间有自己的范围，不同的数根据自己的大小找区间存放，每个区间还设定区间内存放的最小值，最大值
# 最小值按规定放在第一个区间，最大值按规定放在最后一个区间
# 那么也就是说n个数中，只有n-2个数区间未定，要将n-2个数根据每个区间范围放入n-1个区间
# 那么一定至少有一个区间是空闲的，因此也说明，最大间距一定是在不同区间内的两个点之间产生的，也就是下一个区间的最小值-上一个区间的最大值

import math
def maximumGap(nums):
    if len(nums) < 2:
        return 0
    max_val, min_val = max(nums), min(nums)  # 找序列最大值和最小值
    if max_val == min_val:
        return 0
    n = len(nums)
    size = (max_val - min_val) / (n - 1)  # 划分区间

    bucket = [[None, None] for _ in range(n + 1)]
    for num in nums:
        b = bucket[math.floor((num - min_val) // size)]  # 数存放进区间
        b[0] = min(b[0], num) if b[0] else num  # 更新区间最小值
        b[1] = max(b[1], num) if b[1] else num  # 更新区间最大值
    bucket = [b for b in bucket if b[0] is not None]
    return max(bucket[i][0] - bucket[i - 1][1] for i in range(1, len(bucket)))  # 找最大间隙


# nums = [2.3, 3.1, 7.5, 1.5, 6.3]
#
# b = maximumGap(nums)
# print(format(b, '.1f'))


#
# # 找最大值，把数组遍历一遍  O(n)
# def find_max(num, arr):
#     max = 0
#     for i in range(num):
#         if arr[i] >= max:
#             max = arr[i]
#     return max
#
#
# # 找最小值，把数组遍历一遍  O(n)
# def find_min(num, arr):
#     min = arr[0]
#     for i in range(num):
#         if arr[i] < min:
#             min = arr[i]
#     return min
#
#
# def find_maxgap(num, arr):
#     if num < 2:
#         return 0
#     if num == 2:
#         if arr[1]-arr[0] > 0:
#             return arr[1]-arr[0]
#         else:
#             return arr[0] - arr[1]
#     max = find_max(num, arr)
#     min = find_min(num, arr)
#     # 划分为 n-1个区间，将n个数按大小放入n-1个区间。其中规定min放入第1个区间；max放入最后一个区间
#     len_of_gap = (max-min)/(num-1)
#     # 划分好区间之后，将所有数都存放入区间
#     # 每个区间的格式{'min':  ,'max':   ,'count':   }，所有区间存放在一个list_gap列表中
#     list_gap = []
#
#     # 初始化列表
#     for i in range(num-1):
#         single_gap = {'index': i, 'min': max, 'max': min, 'count': 0}  # 这个要放在循环里面
#         list_gap.append(single_gap)
#
#     for i in range(num):  # 遍历每个数
#         if arr[i] == min:
#             # 规定min放入第1个区间
#             # 修改区间中的三个属性值 {'min':  ,'max':   ,'count':   }
#             for term in list_gap:
#                 if term['index'] == 0:
#                     term['min'] = min
#                     term['count'] += 1
#
#         elif arr[i] == max:
#             for term in list_gap:
#                 if term['index'] == num-1-1:
#                     term['max'] = max
#                     term['count'] += 1
#
#         else:
#             index = int((arr[i] - min)//len_of_gap)
#             for term in list_gap:
#                 if term['index'] == index:
#                     if term['min'] > arr[i]:
#                         term['min'] = arr[i]
#                     if term['max'] < arr[i]:
#                         term['max'] = arr[i]
#                     term['count'] += 1
#
#     # 分析：有n-1个区间，而min和max按规定分别存放在第一个区间和最后一个区间，那么也就是说剩下的n-2个数在n-1个区间中存放，
#     # 因此 至少有一个区间会空闲下来，那么也就是说明最大间隙只会在不同区间内的两个数之间产生
#
#     maxgap = 0
#     for term in list_gap:
#         if term['index'] == 0:
#             high = term['max']
#     for i in range(1, num-1):  # 总共有num-1个区间，从第二个区间开始（第二个区间下标为1）
#         if list_gap[i]['count'] != 0:  # 只有当该区间有数据的时候才进行以下步骤
#             tempgap = list_gap[i]['min'] - high
#             if tempgap > maxgap:
#                 maxgap = tempgap
#
#             for term in list_gap:
#                 if term['index'] == i:
#                     high = term['max']
#
#     return maxgap
#
#
# arr = [2.3, 3.1, 7.5, 1.5, 6.3]
#
# num = 5
# # print('最大间距为：', find_maxgap(num, arr))







'''
4、编程实现棋盘覆盖问题

（1）问题描述：在一个2^k * 2^k个方格组成的棋盘中，恰有一个方格与其他方格不同，称该方格为一特殊方格，
且称该棋盘为一特殊棋盘。现在要用L型骨牌覆盖给定的特殊棋盘上除特殊方格以外的所有方格，且任何2个L型骨牌不得重叠覆盖。

（2）算法设计：把棋盘等分成四个正方形分别是：左上、左下、右上、右下 四个子棋盘，对子棋盘进行递归求解。
'''


# 整形二维数组Board表示棋盘，Borad[0][0]使棋盘的左上角方格。
# tile是一个全局整形变量，用来表示L形骨牌的编号，初始值为0。
# tr：棋盘左上角方格的行号；tc：棋盘左上角方格的列号；
# dr：特殊方各所在的行号；dc：特殊方各所在的列号；
# size：s=2n                   
# 将一个棋盘分成四个部分来进行讨论，由于是一个2^k * 2^k个方格，要放L型骨牌，那么一定有一个特殊格子是没法放的
# 讨论，这个特殊格子会在四个区域中的哪个区域
# 对于左上角棋盘：
# 如果特殊格子在左上角区域，那么就直接在左上角区域里面继续进行递归，只要改变棋盘的tr，tc
# 如果不是在左上角，也还是要进行递归，但是由于左上角也还是一个2^k * 2^k的方格，所以一定还会产生一个特殊格子。但是统筹考虑，在一个大棋盘中，分成四个小棋盘，如果对于3个小棋盘的特殊格子都在中间，那么就还能再填入一个L型骨牌
# 所以将左上角棋盘的最右下角的格子规定为特殊格子，再进行递归
# 对于右上角棋盘：
# 如果特殊格子在右上角区域，那么直接在右上角区域里面继续进行递归，只要改变棋盘的tr，tc
# 如果不是再右上角区域，那么右上角棋盘的最左下角的格子规定为特殊格子，所以改变tr、tc、dr、dc的值，继续进行递归
# 以此类推
def CB(tr, tc, dr, dc, size):
    global board
    global tile
    if size == 1:
        return
    tile += 1
    t = tile  # L型骨牌的编号
    s = size // 2  # 分割棋盘
    # 覆盖左上角子棋盘
    if dr < tr+s and dc < tc+s:  # 特殊方格在此棋盘中
        CB(tr, tc, dr, dc, s)
    else:  # 此棋盘中无特殊方格，就用t号L型骨牌覆盖右下角
        board[tr+s-1][tc+s-1] = t
        # 覆盖其余方格
        CB(tr, tc, tr+s-1, tc+s-1, s)

    # 覆盖右上角子棋盘
    if dr < tr+s and dc >= tc+s:  # 特殊方格在此棋盘中
        CB(tr, tc+s, dr, dc, s)
    else:  # 此棋盘中无特殊方格，就用t号L型骨牌覆盖左下角
        board[tr+s-1][tc+s] = t
        # 覆盖其余方格
        CB(tr, tc+s, tr+s-1, tc+s, s)

    # 覆盖左下角子棋盘
    if dr >= tr+s and dc < tc+s:  # 特殊方格在此棋盘中
        CB(tr+s, tc, dr, dc, s)
    else:  # 此棋盘中无特殊方格，就用t号L型骨牌覆盖右上角
        board[tr+s][tc+s-1] = t
        # 覆盖其余方格
        CB(tr+s, tc, tr+s, tc+s-1, s)

    # 覆盖右下角子棋盘
    if dr >= tr+s and dc >= tc+s:  # 特殊方格在此棋盘
        CB(tr+s, tc+s, dr, dc, s)
    else:  # 此棋盘中无特殊方格，就用t号L型骨牌覆盖左上角
        board[tr+s][tc+s] = t
        # 覆盖其余方格
        CB(tr+s, tc+s, tr+s, tc+s, s)


    # 输出矩阵
def show(board):
    n = len(board)
    for i in range(n):
        for j in range(n):
            print(board[i][j], end='	')
        print('')

#
# tile = 0
# n = 8  # 输入8*8的棋盘规格
# board = [[-1 for x in range(n)] for y in range(n)]  # -1代表特殊格子，先全部填上-1
#
# CB(0, 0, 2, 2, n)
# show(board)



'''
5、编程实现最接近点对问题

（1）问题描述：在包含有n个点的集合S中，找出距离最近的两个点。设 p1(x1,y1)，p2(x2,y2)，……，pn(xn,yn)是平面的n个点。
    严格地将，最近点对可能不止一对，输出一对即可。

（2）算法设计：利用分治法思想解决此问题。将集合S分成两个子集S1和S2，最近点对将会出现三种情况：
在S1中，在S2中或者最近点对分别在集合S1和S2中。利用递归法分别计算前两种情况，再与第三种情况相比较，输出三者中最小的距离。
'''
# 但是中间的区域中点可能还是很多，因此继续分析，看能不能减少讨论的点数。
# 对于左半边区域的点p来说，要找到一个右半区域的点q，其之间的距离小于d，那么q只能存在于以p为圆心，以d为半径的圆面和右半边区域相交的区域中
# 经过证明，将那块区域放大，放大成宽为d，长为2d的矩形区域，在这个区域中，要使去其中两个点的距离大于等于d，因此在这块矩形中最多只可能有6个点
# 因此在中间区域讨论时，只需要对中间左半边区域的每个点p，讨论其与中间右半区域中距离他最近的6个点即可
# 那么如何取这6个点呢？
# 首先对中间右半区域的点 进行关于y值的排序，对于p（x,y)来说，这6个点将在右半区域的点中y值∈[y-d,y+d]中间的点中产生
#
#
# 利用分治算法求解
# 将平面中的点按照其横坐标的值，划分成左右两部分，分别对左边部分和右边部分进行递归
# 分别找出左半区域的点对的最短距离和右半区域的点对的最短距离，最终取较小值
# 还有一种情况是 最短距离的点对（p，q）p和q分别在左右半边，因此要对中间区域的点对进行讨论最短距离
# 由于已经找出了左半边和右半边中的最短距离 d
# 因此只需要在中间 宽度为2d的区域内进行讨论即可
# 设 中间x=mid，所以只需要对 左半边x∈[mid-d,mid]  右半边x∈[mid,mid+d]这块中间区域进行讨论即可
# 然后 还要讨论一下特殊情况
# 当只有1个点的时候，返回一个值或报错
# 当只有2个点的时候，返回这两个点之间的距离，结束
# 当有3个点的时候，就计算出这三个点之间的最短距离，结束

import math
import random


# 生成随机点
def generate_point():
    points = []  # 点集
    points_num = random.randint(1, 20)
    for i in range(points_num):
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        points.append([x, y])
    return points


def merge_x(points, start, mid, end):
    temp = []
    i = start
    j = mid + 1
    while i < mid + 1 and j < end + 1:
        if points[i][0] > points[j][0]:  # 根据点的横坐标值进行从小到大排序
            temp.append(points[j])
            j += 1
        else:
            temp.append(points[i])
            i += 1
    while i < mid + 1:
        temp.append(points[i])
        i += 1
    while j < end + 1:
        temp.append(points[j])
        j += 1
    points[start:end + 1] = temp                    # 将排好序的temp赋给points  # 切片是左闭右开，实际不包含第end+1点


# 快速排序  分治
def merge_sort(points, start, end):
    if start < end:
        mid = start + (end - start) // 2
        merge_sort(points, start, mid)
        merge_sort(points, mid + 1, end)
        merge_x(points, start, mid, end)


def distance(p1, p2):  # 求出两个点之间的距离
    x = math.fabs(p1[0] - p2[0])  # 两个点的横坐标距离
    y = math.fabs(p1[1] - p2[1])  # 两个点的纵坐标距离
    return math.sqrt(x*x + y*y)  # 平方和开根号 为距离


# [0, 0, 65535]  【第一个点、第二个点、距离】
# 找最短距离
def get_closest_distance(points, l, r):
    if r <= l:                          # 报错或只有一个点，返回无限大
        return [0, 0, 65535]
    if r - l == 1:                          # 只有两个点直接求解
        return [l, r, distance(points[l], points[r])]
    if r - l == 2:                          # 三个点时使用遍历比较的方法
        d1 = distance(points[l], points[l+1])
        d2 = distance(points[l+1], points[r])
        d3 = distance(points[l], points[r])
        if d1 <= d2 and d1 <= d3:
            return [l, l+1, d1]
        elif d2 <= d1 and d2 <= d3:
            return [l+1, r, d2]
        else:
            return [l, r, d3]
# 四个点以上使用分治法
    m = l + (r - l) // 2  # 划分成左右两半，分治
    result1 = get_closest_distance(points, l, m)
    result2 = get_closest_distance(points, m+1, r)
    d = min(result1[2], result2[2])  # 取左半边和右半边最短距离的较小值
    # 寻找m左侧和右侧范围为d内的所有点  即中间区域内的点
    i = m
    j = m
    while i >= l:                               # 寻找中间区域左侧的值
        if points[m][0] - points[i][0] < d:  # 点的横坐标值的比较来确定点是否在中间左边区域中，横坐标差值小于d就说明在,由于已经关于x坐标值排好序了，那么就只要找到一个界限m就可以
            i -= 1  # 遍历
        else:
            break
    while j <= r:                               # 寻找中间区域右侧的值
        if points[j][0] - points[m][0] < d:  # 点的横坐标值的比较来确定点是否在中间右侧区域中，横坐标差值小于d就说明在
            j += 1  # 遍历
        else:
            break
    if i < m:              # 如果m左侧没有符合条件的点了，此时i会停留在m，如果有，此时i会在最左侧的点之前一位，所以+1
        i += 1
    result3 = [0, 0, 65535]  # 对于中间区域的最短距离[点、点、距离]赋初值

    for a in range(i, m+1):                     # range()左闭右开，我们需要m点作为左侧的点
        for b in range(m+1, j):                 # j此时是右侧中最右边符合条件的点的位置+1，因为range，所以不包含j
            d0 = distance(points[a], points[b])
            if d0 < result3[2]:
                result3 = [a, b, d0]
    # 取最小值
    if result1[2] <= result2[2] and result1[2] <= result3[2]:
        return result1
    elif result2[2] <= result1[2] and result2[2] <= result3[2]:
        return result2
    else:
        return result3


p = generate_point()
print("随机生成点：")
print(p)
merge_sort(p, 0, len(p)-1)
print("归并排序：")
print(p)
result = get_closest_distance(p, 0, len(p)-1)
print("最近的两个点：" + str(p[result[0]]) + ", " + str(p[result[1]]))
print("距离为" + str(result[2]))


#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 求两点间的距离
# def distance(p, i, j):
#     h = pow((pow((p[i][0]-p[j][0]), 2) + pow((p[i][1] - p[j][1]), 2)), 1.0/2)
#     return h
#
#
# def point():
#     n = int(input("点集大小:"))
#     p = [[0, 0]]*n  # 将n个点初始化为（0，0）
#     x = []
#     y = []
#     for i in range(0, n):
#         p[i] = [float(input("x=")), float(input("y="))]  # 人工输入点对
#         x.append(p[i][0])
#         y.append(p[i][1])
#     return p, x, y
#
#
# def cp(pair):
#     global min_i, min_j
#     min_i =[None, None]
#     min_j = [None, None]
#     if len(pair) <= 3:  # 当点数小于等于3个时，就直接计算出他们之间的最短距离
#         length = 1000000  # 找一个很大的数来初始化length，之后length肯定会被替换
#         for i in range(0, len(pair)):
#             for j in range(i+1, len(pair)):
#                 if length >= distance(pair, i, j):  # 找到最短距离
#                     length = distance(pair, i, j)
#                     min_i = pair[i]  # 记录点
#                     min_j = pair[j]  # 记录点
#     else:  # 当点数大于3个时，那么就进行分治
#         mid = len(pair)//2  # 一半的点数
#         x0 = pair[mid][0]  # 找到中间那个点的横坐标
#         lengthl = cp(pair[0:mid+1])  # 左边递归
#         MINI = min_i  # 接收记录点
#         MINJ = min_j  # 接收记录点
#         lengthr = cp(pair[mid+1:])   # 右边递归
#         if lengthl < lengthr:  # 当左边最短距离小于右边最短距离时，将最小点对更新为左边的点对
#             min_i = MINI
#             min_j = MINJ
#         length = min(lengthl, lengthr)  # 将较小的最短距离找出
#         k = 0   # 记录中间左半区域点的个数
#         T = []  # 中间左半区域点的集合
#         for i in range(0, len(pair)):  # 遍历所有点
#             if abs(p[i][0]-x0) <= length:  # 如果x值在[x0-length,x0+length]范围内就入选
#                 T.append(p[i])
#                 k += 1  # 更新点的个数
#         length_ = 2*length
#         for i in range(0, k-1):  # 遍历中间区域的所有点
#             for j in range(i+1, min(i+7, k)):  # 最多比较6个点
#                 if distance(T, i, j) < length_:
#                     length_ = distance(T, i, j)
#                     min_i_ = T[i]
#                     min_j_ = T[j]
#                 if length > length_:  # 如果中间区域点对的最短距离小于左、右半区域的最小距离，那么就让最短点对更新
#                     min_i = min_i_
#                     min_j = min_j_
#         length = min(length, length_)
#     return length
#
# x=[]
# y=[]
# p=[[]]
#p=[]
# for _ in range (0,10):
#     x.append(np.random.randint(-20,20))
#     y.append(np.random.randint(-20,20))
#     p.append([x[_],y[_]])
# p,x,y=point()
# p.sort(key=lambda m:[m[0],m[1]])
# print(p)
# print(cp(p))
# print(min_i)
# print(min_j)
# plt.figure(figsize=(20,8),dpi=80)
# plt.scatter(x,y,s=20,c='b',marker='*')
# plt.plot([min_i[0],min_j[0]],[min_i[1],min_j[1]],c='r')
# plt.show()






