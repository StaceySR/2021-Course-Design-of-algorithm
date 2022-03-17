# 1、用Python语言编程实现二分搜索算法：
# 已知不重复且已经按从小到大排好的m个整数的数组A[1..m]（设m=2 k，k是一个确定的非负整数）。对于给定的整数c，要求寻找一个下标i，使得A[i]=c；若找不到，则返回一个0。
def bi_search(array):
    low = 0
    high = len(array) - 1
    x = input("请输入你要查找其下标的数：")  # x为字符串类型
    y = int(x)
    while low <= high:
        mid = (low + high) // 2
        if y == array[mid]:
            return "该数所在下标为{}".format(mid)
        elif y > array[mid]:
            low = mid + 1
        else:
            high = mid - 1
    return 0


def main():
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = bi_search(a)
    print(b)


if __name__ == "__main__":
    main()


# # 2、用Python语言编程实现至少7种排序算法。
# # ---------各大排序算法--------------------
#
#
# # # 1、冒泡排序：每次冒泡，将最大的元素冒到最后面；第一次是前n个元素，最大的元素冒到最后面，第二次是前n-1个元素，最大的元素冒到倒数第二个位置
# def bubble_sort(nums):
#     n = len(nums)
#     # 从第一个数开始，第一个数与第二个，第二个与第三个，进行比较，每次比较都把较大的往后移，知道全部比完，最大的数就在最后了
#     # 那第二次开始，就不用去管最后那个已经是最大的数了，只要去找前面n-1个中最大的数即可
#     for i in range(n - 1):  # 75234   # 0   #1    #2   #3
#         for j in range(n - i - 1):  # 0123  #012  #01  #0
#             if nums[j] > nums[j + 1]:  # 从第0个数开始，每次与后面相邻以为比较，把较大的往后移，直到最大的数被放到最后
#                 nums[j], nums[j + 1] = nums[j + 1], nums[j]
#         # 第一次循环结束，最大的数已经排在最后，接下来只需要对前面还没排好序的数进行相同操作，每次都把最大的数放到最后
#     return nums
#
#
# # 2、快速排序:在一遍快速排序中，以基准值为基础，将比基准值小的数放到基准值的左边，比基准值大的数放到基准值的右边。
# #            然后递归，分别使基准值左边和右边的数快速排序
# #  （1）遍历一遍数组，用两个空的数组来存储比基准值大和比基准值小的数，会使用额外的空间
# def quick_sort1(nums):
#     n = len(nums)
#     if n == 1 or n == 0:  # 只有1个或0个数时不需要排序
#         return nums
#     left = []  # left列表来装比基准值小的数
#     right = []  # right列表来装比基准值大的数
#     for i in range(1, n):
#         if nums[i] <= nums[0]:  # 设定基准值为第一个数
#             left.append(nums[i])  # 大于等于基准值的放入left列表
#         else:
#             right.append(nums[i])  # 小于基准值的放入right列表
#     # 之后分别就是递归，分别对left,right列表实行快速排序
#     return quick_sort1(left) + [nums[0]] + quick_sort1(right)  # 注意nums[0]这个元素要放在一个列表中
#
#
# # # 3、
# # def quick_sort2(nums, start, end):
# #     if start >= end:  # 递归的退出条件
# #         return
# #     mid = nums[start]  # 设定起始的基准元素
# #     low = start  # low为序列左边在开始位置的有座享有移动的游标
# #     high = end  # high为序列右边末尾位置的由右向左移动的游标
# #     while low < high:
# #         # 如果low与high未重合，high（右边）指向的元素大于等于基准元素，则high向左移动
# #         while low < high and nums[high] >= mid:
# #             high -= 1
# #         nums[low] = nums[high]  # 走到此位置时high指向一个比基准元素小的元素，将high指向的元素放到low的位置上，此时high指向的位置空着，接下来移动low找到符合条件的元素放在此处
# #         # 如果low与high为重合，low指向的元素比基准元素小，则low向右移动
# #         while low < high and nums[low] < mid:
# #             low += 1
# #         nums[high] = nums[low]  # 此时low指向一个比基准元素大的元素，将low指向的元素放到high空着的位置上，此时low指向的位置空着，之后进行下一次循环，将high找到符合条件的元素填到此位置
# #
# #     # 退出循环后，low与high重合，此时所指位置为基准元素的正确位置，左边的元素都比基准元素小，右边的元素都比基准元素大
# #     nums[low] = mid  # 将基准元素放到该位置
# #     # 对基准元素右边的子序列进行快速排序
# #     quick_sort2(nums, start, low - 1)  # start：0  low：-1 原基准元素靠左边一位
# #     # 对基准元素右边的子序列进行快速排序
# #     quick_sort2(nums, low + 1, end)  # low+1：原基准元素靠右一位  end：最后
#
#
# #  4、简单插入排序：将某一个数字插入到已经排好序的数组当中
# def insert_sort(nums):
#     n = len(nums)
#     for i in range(1, n):  # 75234     # 1     # 2              #3             #4
#         index = i                    #57234   #52734，25734 #25374，23574  #23547，23457
#         for j in range(i - 1, -1, -1):  # 0   # 10             #210           #3210
#             if nums[j] > nums[index]:
#                 # 第2个数，第3个数。。。每个数都要与其前面的数进行比较，如果比前面的数小，那么就要插入到该数之前。
#                 # 每次都是j=index-1。如果nums[j] > nums[index]成立，那么就将nums[j]与nums[index]交换位置，并将index-1。其实就相当于将nums[index]插到nums[j]的前面。
#                 nums[index], nums[j] = nums[j], nums[index]
#                 index -= 1
#             else:
#                 break
#     return nums
#
#
# # def insert_sort_2(nums):
# #     for i in range(1, len(nums)):
# #         key = nums[i]
# #         j = i - 1
# #         while j >= 0:
# #             if nums[j] > key:
# #                 nums[j + 1] = nums[j]
# #                 nums[j] = key
# #             j -= 1
# #     return nums
#
#
# def shell_sort2(alist):
#     n = len(alist)
#     gap = n//2  # 分组，在每组中，各自进行插入排序，随着gap的不断缩小，整体数据渐趋有序，插入排序的效率也越来越高。当最后gap=1时，就是普通的插入排序，但是因为数据已经大致有序所以效率很高。0,0+gap,0+gap+gap.....|1,1+gap,1+gap+gap....|2,....
#     while gap > 0:
#         for i in range(gap, n):
#             while i > 0:
#                 if alist[i] < alist[i-gap]:
#                     alist[i], alist[i-gap] = alist[i-gap], alist[i]
#                     i = i - gap
#                 else:
#                     break
#         gap //= 2
#     return alist
#
# # 5、希尔排序
# # def shell_sort(nums):
# #     n = len(nums)  # 75234   #5
# #     step = 2
# #     group = n // step  #5//2=2
# #     while group > 0:
# #         for i in range(group):  #0     #1
# #             j = i + group      #2      #3
# #             while j < n:
# #                 k = j - group  #0      #1
# #                 key = nums[j]  #2     #3
# #                 while k >= 0:
# #                     if nums[k] > key:
# #                         nums[k + group] = nums[k]
# #                         nums[k] = key
# #                     k -= group
# #                 j += group
# #         group //= step
# #     return nums
#
#
#
#
# print(shell_sort2([7, 3, 0, 0, 4, 5, 1, 1]))
#
#
# # 6、简单选择排序：每一次选择一个当前最小的元素放在已经排好序的数组后面
# def head_sort(nums):
#     n = len(nums)
#     for i in range(n): # 从第一个数开始遍历          #75234      #0        #1             #2          #3       #4
#         index = i                           #25734        23754         23457    2457
#         for j in range(i + 1, n):  # 找出当前遍历到的第i+1个数及其后面的数中最小的数   #1234                         #234             #34             #4
#             if nums[j] < nums[index]:
#                 index = j
#         # 跳出循环之后，index中记录着当前遍历到的第i+1个数及其之后的数中最小的数的下标
#         # 将找到的最小的数放到当前遍历到的第i+1个数的位置。也即放在之前排好序的数的后面
#         nums[i], nums[index] = nums[index], nums[i]
#     return nums
#
#
#
# def head_sort_2(nums):
#     for i in range(len(nums) - 1, 0, -1):  # 从最后一个数开始往前遍历# 75234      #4     #3             #2             #1
#         for j in range(i):                           #0123   #012           #01             #0
#             if nums[j] > nums[i]:    # 找出当前遍历到的第i+1个数及其之前的i个数之前最大的数，放在第i+1个数的位置。也即放在之前排好序的数的前面        #45237         #35247 34257   #24357 23457     #23457
#                 nums[j], nums[i] = nums[i], nums[j]
#     return nums
#
#
# # 7、-----------堆排序：构造最大堆，把最上面那个节点一道最后一个节点去
# def sift_down(nums, start, end):
#     root = start
#     while True:
#         # 从root开始对最大堆调整
#
#         # 1、找左孩子节点
#         child = 2 * root + 1
#         if child > end:
#             break
#
#         # 2、找出两个child中较大的一个，最后只需把较大的child与root比较即可
#         if child + 1 <= end and nums[child] < nums[child + 1]:
#             child += 1
#
#         # 3、root与child的值进行比较，保证root更大
#         if nums[root] < nums[child]:
#             nums[root], nums[child] = nums[child], nums[root]
#             # ------调整过后，也要保证下面的堆仍然满足大根堆
#             # 即把刚刚更改过的child作为节点，等着上面的步骤 4、再循环，保证child下面的对也是大根堆
#             root = child
#         else:
#             # 没有调整，推出
#             break
#
#
# def heap_sort(nums):
#     # 从最下层的最右端的根节点开始调整大根堆,然后是倒数第二个根节点，倒数第三个。。
#     first = len(nums) // 2 - 1
#     for start in range(first, -1, -1):
#         sift_down(nums, start, len(nums) - 1)
#
#     # 将最大的放到堆最后一个，堆-1，继续调整排序
#     for end in range(len(nums) - 1, 0, -1):
#         nums[0], nums[end] = nums[end], nums[0]
#         sift_down(nums, 0, end - 1)  # 每次把最顶端的根节点换下去之后 start就是0。由于找到一个最大值把他移到最后面，就只要对剩下的继续找出最大值，所以所有节点数-1，即end-1
#
#     return nums
#
#
# def main():
#     nums = [4, 5, 3, 7, 2, 6, 8, 1, 9, 0]
#     print(bubble_sort(nums))
#     print(quick_sort1(nums))
#     quick_sort2(nums, 0, len(nums) - 1)
#     print(nums)
#     print(insert_sort(nums))
#     print(shell_sort(nums))
#     print(head_sort(nums))
#     print(heap_sort(nums))
#
#
# if __name__ == "__main__":
#     main()
