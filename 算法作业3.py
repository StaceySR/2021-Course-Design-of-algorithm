
    #1、二分查找问题：在一个无重复的有序整数数组中查找某个数的位置，如果找到则返回下标，否则返回-1。

def binary_search(arr, num): #  1 4 5 6  二分查找的思想就是将num与有序整数组的中间那个数比较大小，若相等，就直接找到了下标。
    # 若num更大，就直接排除掉有序整数组的左半部分数据
    # 若num更小，就直接排除掉有序整数组的右半部分数据
    # 每一次都排除掉一半的数据，在剩下的一半中继续排除一半，递归下去，直到左边界>右边界，退出循环
    left = 0
    right = len(arr) - 1
    # 其实有序数组分为 升序 降序 两种，前面讨论的时升序情况，降序的正好相反
    # 那就需要首先判断有序数组时升序还是降序
    if len(arr) == 0:  # 数组含有0个元素时
        return -1
    elif len(arr) == 1:  # 数组含有1个元素时，只有两种情况。1、相等，即下标为0  2、不等，即返回-1
        if num == arr[0]:
            return 0
        else:
            return -1
    else:
        # 从数组个数大于等于2，开始判断有序数组时升序还是降序
        if arr[0] < arr[1]:  # 当第一个数小于第二个数，则升序
            while left <= right:
                now = (left + right) // 2  # 定位有序整数组中间数据的下标
                if num == arr[now]:
                    return now
                elif num < arr[now]:
                    right = now - 1
                else:
                    left = now + 1
            # 当退出循环时还没有return，就代表找不到下标，要返回-1
            return -1
        else:  # 降序
            while left <= right:
                now = (left + right) // 2  # 定位有序整数组中间数据的下标
                if num == arr[now]:
                    return now
                elif num < arr[now]:
                    left = now + 1
                else:
                    right = now - 1
            # 当退出循环时还没有return，就代表找不到下标，要返回-1
            return -1


# arr = [99, 88, 77, 6, 5, 4, 3, 2, 1]
# print(binary_search(arr, 99))


'''
    2、 有重复的二分查找：在一个可重复的升序的整数数组中查找某个数的开始位置和结束位置。 如果数组中不存在，则返回 [-1, -1]。 算法时间复杂度要求为 O(log n) 。
    示例1：
    输入：nums = [5,7,7,8,8,10]
               target = 8
    输出：[3,4]
    示例2：
    输入：nums = [5,7,7,8,8,10]
               target = 6
    输出：[-1,-1]
'''


def re_binary_search(nums, target):
    left = 0
    right = len(nums) - 1
    if len(nums) == 0:  # 数组含有0个元素时,直接返回[-1,-1]
        return [-1, -1]
    elif len(nums) == 1:  # 数组含有1个元素时，只有两种情况。1、相等，即开始位置为0，结束位置为0  2、不等，即返回[-1,-1]
        if target == nums[0]:
            return [0, 0]
        else:
            return [-1, -1]
    while left <= right:  # 数组含有2个及以上元素时
        now = (left+right) // 2  # now定位数组中间的数据项
        if target == nums[now]:  # 当target恰好等于升序数组中间那个数时，以left作为其开始位置，right作为其结束位置
            left = now  # 以防这个数没有重复项，那开始位置和结束位置都是now，所以就先给left和right赋上now
            right = now
            keyl = now - 1  # keyl为中间偏左一个数
            keyr = now + 1  # keyr为中间偏右一个数
            while keyl >= 0 and nums[keyl] == target:  # 这个循环是判断 now左边的数据有没有和target重复的，如果有，就要将left往前移！ 注意这里需要控制keyr在下标范围内
                left = keyl
                keyl -= 1  # 如果有重复，就不断往前查看
            while keyr <= len(nums)-1 and nums[keyr] == target:  # 这个循环是判断now右边的数据有没有和target重复的，如果有，就要将right往后移！ 注意这里需要控制keyr在下标范围内
                right = keyr
                keyr += 1  # 如果有重复，就不断往后查看
            return [left, right]
        elif target > nums[now]:  # 当目标比当前数大时，直接排除掉左边一部分，仅看右边一部分
            left = now + 1
        else:  # 当目标比当前数小时，直接排除掉右边一部分，仅看左边一部分
            right = now - 1

    return [-1, -1]  # 退出循环时还没有结束程序，说明该数组中找不到该数，就返回[-1,-1]


# nums = [3, 4, 4, 5, 6]
# print(re_binary_search(nums, 6))



'''
    3、矩阵查找问题：在一个 m x n 的有序整数矩阵中查找某个数。有序矩阵是指每行中的整数从左到右为升序排列、每行的第一个整数大于前一行的最后一个整数。找到返回true，否则返回false。
    示例1：
    输入：matrix = [ [1,  3,  5,  7],
                    [10, 11, 16, 20],
                    [23, 30, 34, 50]] 
               target = 3
    输出：true
    示例2：
    输入：matrix = [ [1,  3,  5,  7],
                    [10, 11, 16, 20], 
                    [23, 30, 34, 50]] 
              target = 13
    输出：false
'''


def matrix_search(matrix, target):  # len(matrix) 为矩阵行数，len(matrix[0]) 为矩阵列数
    row = 0
    column = len(matrix[0]) - 1  # 每次都从矩阵每行的最后一个数开始与target比较。
                                # 比较结果可能产生三种情况：1、与target相等，则直接返回True
                                                       # 2、target较大，则直接行数加1，列数不变，从下一行的最后一个数开始同样的方法比较
                                                       # 3、target较小，则寻找范围缩小到该行，行数不变，列数-1开始遍历该行
    while row <= len(matrix)-1 and column >= 0:  # 只在行数、列数范围内进行查找
        if target == matrix[row][column]:  # 与target相等，则直接返回True
            return True
        elif target > matrix[row][column]:  # 如果target较大，那么直接行数加1，列数不变，将下一行的最后一个数与target比较，直接排除掉了一行
            row += 1
        else:  # 如果target较小，那么就说明在target要么是在该行中，要么就不存在。所以首先把该行遍历完，column逐渐-1，遍历到column<0时，就退出while循环，那么target就是不存在，则返回False
            column -= 1
    return False  # 退出循环还没有返回时，就说明找不到该值，返回False


matrix = [[1, 3, 5, 7],
          [10, 11, 16, 20],
          [23, 30, 34, 50]]


# target = 3
# print('行数：', len(matrix))
# print('列数：', len(matrix[0]))
# print(matrix_search(matrix, target))


'''
    4、 寻找两个有序数组的中位数：给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。 找出这两个有序数组的中位数， 假设 nums1 和 nums2 不会同时为空，要求算法的时间复杂度为 O(log(m + n))。 
    示例1：输入：nums1 = [1, 3]           nums2 = [2]          输出：2.0
    示例2：输入：nums1 = [1, 2]           nums2 = [3, 4]       输出：2.5
'''
# 中位数，又称中点数，中值。中数是按顺序排列的一组数据中居于中间位置的数，即在这组数据中，有一半的数据比他大，有一半的数据比他小。
# 也就是说，中位数把一个集合划分为长度相等的两个子集，一个子集的元素问题大于另一个子集。


#  这个题目可以归结到寻找第k小(大)元素问题，
#  思路可以总结如下：取两个数组中的第k/2个元素进行比较，
#  如果数组1的元素小于数组2的元素，则说明数组1中的前k/2个元素
#  不可能成为第k个元素的候选，所以将数组1中的前k/2个元素去掉，
#  组成新数组和数组2求第k-k/2小的元素，因为我们把前k/2个元素
#  去掉了，所以相应的k值也应该减小。另外就是注意处理一些边界
#  条件问题，比如某一个数组可能为空或者k为1的情况。

# 由于题目规定说要求算法的时间复杂度为 O(log(m + n))，所以想到二分
# 找中位数也就是找最中间那个数。如果最中间有两个数，则求他们的平均数
# 假设len1=len（nums1） len2=len（nums2）
    # 如果len1+len2为奇数，那么最中间那个数就是 （len1+len2+1）//2；
    # 如果len1+len2为偶数，那么最中间两个数就是 （len1+len2+1）//2和（len1+len2+2）//2

def midnum_search(nums1, nums2):
    def findKth(arr1, arr2, k):  # 找第k小的数
        if len(arr1) == 0:  # 当数组1为空，而数组2不为空时，那么就只需找到已经排好序的数组二的第k小的数即可
            return arr2[k-1]
        if len(arr2) == 0:  # 当数组2为空，而数组1不为空时，那么就只需找到已经排好序的数组一的第k小的数即可
            return arr1[k-1]
        if k == 1:  # 当k=1时，就是找第1小的数，因为arr1与arr2都是升序的，那么就只要找到两个数组第一个数较小的那个即可
            return min(arr1[0], arr2[0])
        i = min(k//2, len(arr1)) - 1  # 这里体现二分思想，找出第k//2个数。但是要注意数组的长度可能小于k//2。当小于时，就将i赋为数组最后一个数的下标
        j = min(k//2, len(arr2)) - 1
        if arr1[i] > arr2[j]:  # 当arr1的k//2个数大于arr2的k//2个数，那么说明arr[0:j]一定时在第k个数之前的，所以不可能时第k个数的候选，因此直接取arr[k+1:]继续进行递归。
            return findKth(arr1, arr2[j+1:], k-(j+1))  # 注意这里的k的值也要跟着改变：由于已经排除了第k个之前的j+1个数，那么在剩下的数中继续找原来的第k个数的话，k要变成k-(j+1)
        else:
            return findKth(arr1[i+1:], arr2, k-(i+1))
    len1 = len(nums1)
    len2 = len(nums2)
    if len1 == 0 and len2 == 0:  # 如果两个数组都为空，那么无法找到中位数 。并且能够保证进入findKth函数的数组最多一个为空
        return '两个空数组找不到中位数'
    return (findKth(nums1, nums2, (len1+len2+1)//2) + findKth(nums1, nums2, (len1+len2+2)//2))/2  # 这里把len1+len2为奇数和偶数都一起考虑了，如果是奇数，那么(len1+len2+1)//2与(len1+len2+2)//2是一样的


nums1 = [1, 2]
nums2 = [3, 4]
print(midnum_search(nums1, nums2))




'''
    5、 假设给定的一个数组描述了 n 个宽度为 1 的柱子高度，请求出按此排列的柱子，下雨之后能接多少雨水。
    示例：
    输入： [0,1,0,2,1,0,1,3,2,1,2,1]
    输出： 6
    示例图：

'''
def rain(arr):
    left = 0  # 每一个小蓄水坑的左边界（蓄水坑只有呈凹槽状才有蓄水能力）
    right = 1  # 每一个小蓄水坑的右边界（蓄水坑只有呈凹槽状才有蓄水能力）
    v = 0  # 累加记录蓄水量
    while left <= right and right <= len(arr)-1:  # 边界
        if arr[left] > arr[right]:  # 只有左边比右边高，呈下降趋势，才需要进入以下讨论，不然肯定没办法蓄水
            while right < len(arr) - 1 and arr[right] >= arr[right+1]:  # 当下降阶梯一直存在时，右边界就一直往后移。！！注意，等号必须并在下降状态时，不能并在下面上升状态，不然会影响检验是否有上升状态
                right += 1
            while right < len(arr) - 1 and arr[right] < arr[right + 1]:  # 结束下降状态，开始上升时，当上升阶梯一直存在时，右边界继续往后移
                right += 1
            if arr[right] > arr[right - 1]:  # 检验是否有上升状态。因为必须要有上升状态，才能蓄水，才需要进入以下讨论改变v的值
                if arr[left] >= arr[right]:  # 当左边界的值大于右边界，那么就用 较短边界---右边界 来做限制
                    k = left + 1
                    while k < right:
                        v += (arr[right] - arr[k])  # 把凹槽的蓄水量都加起来
                        k += 1
                else:  # 当左边界的值小于右边界时，那么就用 较短边界---左边界 来做限制
                    k = left + 1
                    while k < right:  # ！注意，这里k绝对不能等于right。因为当k=right，arr[left] < arr[right]时，arr[left] - arr[right] < 0
                        v = v + (arr[left] - arr[k])  # 把凹槽的蓄水量都加起来
                        k += 1
        left = right  # 当arr[left] <= arr[right]时，就说明没有下降趋势，那么就不必进入上述讨论，因为肯定没有蓄水能力。
        right += 1  # 直接将left和right往后遍历
    return v

#
# arr = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
# print(rain(arr))















