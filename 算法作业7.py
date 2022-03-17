''''''



'''
1、世界地图上相邻国家需要用不同的颜色标注以示区别，但最多只需要选取四种颜色即可。
请编程实现图的着色问题。
'''


def four_color_map(map):
    Num = len(map)  # 地图上国家个数
    Color = [0 for i in range(Num)]  # 每个国家的颜色，初始化为0
    n = m = 1  # m代表第m个国家m∈[1,Num]，n代表第n号颜色n∈[1,4]
    # 染色第一个区域，先设置为1
    while m <= Num:
        while n <= 4 and m <= Num:
            flag = True
            for k in range(m - 1):  # 第m个国家染上n号颜色的时候，查看从第1个到第m-1个国家有没有与第m个国家相邻，并且颜色也是n号，如果有，那么就冲突了
                if map[m - 1][k] == 1 and Color[k] == n:
                    flag = False  # 染色有冲突
                    n += 1   # 换下一种颜色尝试
                    break
            if flag:  # 说明染色没冲突
                Color[m - 1] = n  # 给第m个国家记录下n号颜色
                m += 1  # 开始为下一个国家染色
                n = 1  # 颜色从1号开始尝试
        if n > 4:  # 超出标记范围 必须回退
            m -= 1  # 修改前一个国家的颜色
            n = Color[m - 1] + 1  # 让前一个国家从下一个颜色开始试
    return Color


# 两点相邻为1，两点不相邻为0
map1 = [
    [0, 1, 0, 0, 0, 0, 1],  # 第1个国家与2、7个相邻
    [1, 0, 1, 1, 1, 1, 1],  # 第2个国家与1、3、4、5、6、7个相邻
    [0, 1, 0, 1, 0, 0, 0],  # 第3个国家与2、3个相邻
    [0, 1, 1, 0, 1, 0, 0],  # 第4个国家与2、3、5个相邻
    [0, 1, 0, 1, 0, 1, 0],  # 第5个国家与2、4、6个相邻
    [0, 1, 0, 0, 1, 0, 1],  # 第6个国家与2、5、7个相邻
    [1, 1, 0, 0, 0, 1, 0]   # 第7个国家与1、2、6个相邻
]

map2 = [
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
]

map3 = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]

map_color1 = four_color_map(map1)
map_color2 = four_color_map(map2)
map_color3 = four_color_map(map3)
for i in range(len(map_color1)):
    print('第%d个国家的颜色是:' % int(i+1), '%d号' % map_color1[i])

print('--------------------------------')

for i in range(len(map_color2)):
    print('第%d个国家的颜色是:' % int(i+1), '%d号' % map_color2[i])

print('--------------------------------')

for i in range(len(map_color3)):
    print('第%d个国家的颜色是:' % int(i+1), '%d号' % map_color3[i])



'''
2、编程实现0-1背包问题，至少两种算法。
'''


# 1、动态规划方法求解
def knapsack_1(value_list, weight_list, total_weight, total_length):
    # 初始化记录矩阵，当一个物品都不选择时，result[0][total_weight]肯定为0；当总重量限制为0时，result[i][0]肯定也为0.
    result = [[0 for j in range(total_weight+1)] for i in range(total_length+1)]
    # result[i][j]表示到第i个元素为止，在限制总重量为j的情况下能得到的最优解
    # 这个最优解要么包含第i个物品，要么不包含
    for i in range(1, total_length+1):
        for j in range(1, total_weight+1):
            if weight_list[i] <= j:  # 如果第i个物品还在j范围内
                # 那么就意味着可以选择它，但是要不要选择第i个物品呢
                # 比较 1、选择了第i个物品的结果result[i-1][j-weight_list[i]]+value_list[i] 2、不选择第i个物品的结果result[i-1][j]  ，选择较大者
                result[i][j] = max(result[i-1][j-weight_list[i]]+value_list[i], result[i-1][j])
            else:  # 第i个物品不在j范围内
                result[i][j] = result[i-1][j]
    for i in range(len(result)):
        print(result[i])
    return result[-1][-1]


if __name__ == '__main__':
    v1 = [0,60,100,120]
    w1 = [0,10,20,30]
    weight1 = 20
    n1 = 3
    result1 = knapsack_1(v1, w1, weight1, n1)
    print("最优解为：", result1)

    v2 = [0,10,6,3,7,2]
    w2 = [0,6,2,2,5,1]
    weight2 = 7
    n2 = 5
    result2 = knapsack_1(v2, w2, weight2, n2)
    print("最优解为：", result2)


# 2、回溯法
best_value = 0  # 最大价值
now_weight = 0  # 当前背包重量
now_value = 0  # 当前背包价值
best_path = None  # 最优解路径


def backtrack(i):
    global best_value, best_path, now_value, now_weight, path
    if i >= n:
        if best_value < now_value:
            best_value = now_value
            best_path = path[:]
    else:
        if now_weight + weight[i] <= c:  # c为背包总质量，放入背包中物品的总质量小于等于背包容量
            path[i] = 1  # 为1时相当于要第i个物品，走左边分支
            now_weight += weight[i]
            now_value += value[i]
            backtrack(i+1)  # 在解空间中继续往下层走，搜到某一步时，发现不是最优或者达不到目标，则退一步重新选择
            now_weight -= weight[i]  # 回溯，即将原来加上去的重量与价值恢复
            now_value -= value[i]
        path[i] = 0  # 为0时相当于不要第i个物品，走右边分支
        backtrack(i+1)  # 换一条路走


if __name__ == '__main__':
    n = 8
    c = 13  # c为背包总质量
    weight = [1,5,3,4,2,2,1,4]
    value = [2,3,1,5,7,3,2,1]
    path = [0 for i in range(n)]
    backtrack(0)
    print('最大价值: ', best_value)  # 最大价值
    print('在解空间中走的路径: ', best_path)  # 在解空间中走的路径
    choose_box = []  # 最终被选择的哪些物品
    for i in range(n):
        if best_path[i] == 1:
            choose_box.append(i+1)
    print('最终被选择的哪些物品: ', choose_box)  # 构成最优解时，被选择的物品




'''
3、给定一个二维网格和一个单词，找出该单词是否存在于网格中。
说明：单词必须按照字母顺序，通过相邻的单元格内的字母构成，
其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。
同一个单元格内的字母不允许被重复使用。

示例：board = [ ['A','B','C','E'], 
                ['S','F','C','S'],
                ['A','D','E','E']] ，给定 word = "ABCCED", 返回 true， 给定 word = "SEE", 返回 true， 给定 word = "ABCB", 返回 false。
'''

# 给定一个二维网格和一个单词，找出该单词是否存在于网格中。
# 单词必须按照字母顺序，通过相邻的单元格内的字母构成，
# 其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。
# 同一个单元格内的字母不允许被重复使用。


def exist(board, word):
    if len(board) == 0:
        return False

    rows = len(board)
    cols = len(board[0])

    # 这里用以标记二维网格中的元素是否使用（由题意，使用过的不能再使用）
    # False 表示未使用，初始化都为未使用
    # True 表示已使用
    marked = [[False for _ in range(cols)] for _ in range(rows)]

    # 遍历二维网格
    for row in range(rows):
        for col in range(cols):
            # 利用search函数去搜索该单词，当找到所有元素时返回 True，意味着单词存在，“0”代表以搜索到0个元素
            if search(row, col, board, word, 0, marked):
                return True
    return False


def search(i, j, board, word, index, marked):
    # 当搜索到word中的所有元素，就终止
    if index == len(word) - 1:
        return board[i][j] == word[index]

    # 只有当前二维表位置的字母与单词中的字母匹配了才继续搜索
    if board[i][j] == word[index]:
        # 这里先标记元素，如果搜索不成功的情况下，解除标记
        marked[i][j] = True

        directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
                    #  向下、   向左、    向上、  向右

        # 四个方位搜索
        for dx, dy in directions:
            nrow = i + dx
            ncol = j + dy

            # 在边界内、找相邻未使用过的元素、且能走得通的
            if 0 <= nrow < len(board) and 0 <= ncol < len(board[0]) and not marked[nrow][ncol] and search(nrow, ncol, board, word, index+1, marked):
                return True
        # 搜索不成功，标记改为False
        marked[i][j] = False
    return False


board = [['A', 'B', 'C', 'E'],
         ['S', 'F', 'C', 'S'],
         ['A', 'D', 'E', 'E']]
word1 = "ABCCED"
word2 = "SEE"
word3 = "ABCB"
print(exist(board, word1))
print(exist(board, word2))
print(exist(board, word3))


'''
4、班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。
如果已知 A 是 B 的朋友，B 是 C 的朋友，那么我们可以认为 A 也是 C 的朋友。
所谓的朋友圈，是指所有朋友的集合。给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。
如果M[i][j] = 1，表示已知第 i 个和 j 个学生互为朋友关系，否则为不知道。
你必须输出所有学生中的已知的朋友圈总数。

示例1：输入: [[1,1,0], [1,1,0], [0,0,1]]       输出: 2      
说明：已知学生0和学生1互为朋友，他们在一个朋友圈。第2个学生自己在一个朋友圈。所以返回2。

示例2：输入: [[1,1,0],   [1,1,1],  [0,1,1]]       输出: 1      
说明：已知学生0和学生1互为朋友，学生1和学生2互为朋友，所以学生0和学生2也是朋友，所以他们三个在一个朋友圈，返回1。

'''

#
# def findCircleNum(M):
#     def dfs(i):
#         for j in range(n):
#             if M[i][j]:
#                 M[i][j] = M[j][i] = 0
#                 dfs(j)
#
#     n = len(M)
#     friend_cicle = 0
#     for i in range(n):
#         if M[i][i]:  # 一开始先将一个人算一个朋友圈，然后通过dfs去查看有没有跟他同一个朋友圈的，如果有，那么就将那位相关位上的值赋为0；同理递归再去找有没有和这个朋友在同一个朋友圈的
#             M[i][i] = 0
#             friend_cicle += 1
#             dfs(i)
#     return friend_cicle
#


def findCircleNum(M):
    def dfs(M, visited, i):
        nums = M[i]
        for j in range(len(M[0])):
            if visited[j] == 0 and nums[j] == 1:
                visited[j] = 1
                dfs(M, visited, j)
    visited = [0 for i in range(len(M))]  # 访问记录
    count = 0
    for i in range(len(M)):
        if visited[i] == 0:
            count += 1
            dfs(M, visited, i)
    return count


M1 = [[1, 1, 0],
      [1, 1, 0],
      [0, 0, 1]]

M2 = [[1, 1, 0],
      [1, 1, 1],
      [0, 1, 1]]

M3 = [[1, 0, 1, 0],
      [0, 1, 0, 0],
      [1, 0, 1, 1],
      [0, 0, 1, 1]]

print('朋友圈总数：', findCircleNum(M1))
print('朋友圈总数：', findCircleNum(M2))
print('朋友圈总数：', findCircleNum(M3))



'''
5、你有一个用于表示一片土地的整数矩阵land，该矩阵中每个点的值代表对应地点的海拔高度。
若值为0则表示水域。由垂直、水平或对角连接的水域为池塘。池塘的大小是指相连接的水域的个数。
编写一个方法来计算矩阵中所有池塘的大小，返回值需要从小到大排序。

示例：输入  [[0,2,1,0],
            [0,1,0,1],  
            [1,1,0,1],  
            [0,1,0,1]]      
      输出： [1,2,4]
'''


def pondSizes(land):
    m = len(land)
    n = len(land[0])
    # 8个方向
    direction = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [-1, -1], [-1, 1], [1, -1]]
    pond = []

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n:
            return 0
        if land[i][j] != 0:
            return 0
        land[i][j] = 1
        pond_size = 1  # 此时已经能确定池塘里面有一个水域
        for d in direction:  # 往8个方向去找
            pond_size += dfs(i + d[0], j + d[1])
        return pond_size

    for i in range(m):
        for j in range(n):
            if land[i][j] == 0:
                pond.append(dfs(i, j))  # 已找到一个池塘，添加进池塘列表
    return sorted(pond)  # 排序


land1 = [[0, 2, 1, 0],
         [0, 1, 0, 1],
         [1, 1, 0, 1],
         [0, 1, 0, 1]]

land2 = [[1, 1, 0],
         [1, 1, 1],
         [0, 1, 1]]

land3 = [[1, 0, 1, 0],
         [0, 1, 0, 0],
         [1, 0, 1, 1],
         [0, 0, 1, 1]]

print(pondSizes(land1))
print(pondSizes(land2))
print(pondSizes(land3))


