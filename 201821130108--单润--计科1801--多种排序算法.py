 # 1、冒泡排序
def bubble_sort(nums):
    n = len(nums)
    # 从第一个数开始，第一个数与第二个，第二个与第三个，进行比较，每次比较都把较大的往后移，知道全部比完，最大的数就在最后了
    # 那第二次开始，就不用去管最后那个已经是最大的数了，只要去找前面n-1个中最大的数即可
    for i in range(n - 1):  # 75234   # 0   #1    #2   #3
        for j in range(n - i - 1):  # 0123  #012  #01  #0
            if nums[j] > nums[j + 1]:  # 从第0个数开始，每次与后面相邻以为比较，把较大的往后移，直到最大的数被放到最后
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
        # 第一次循环结束，最大的数已经排在最后，接下来只需要对前面还没排好序的数进行相同操作，每次都把最大的数放到最后
    return nums


# 2、快速排序
def quick_sort(nums):
    n = len(nums)
    if n == 1 or n == 0:  # 只有1个或0个数时不需要排序
        return nums
    left = []  # left列表来装比基准值小的数
    right = []  # right列表来装比基准值大的数
    for i in range(1, n):
        if nums[i] <= nums[0]:  # 设定基准值为第一个数
            left.append(nums[i])  # 大于等于基准值的放入left列表
        else:
            right.append(nums[i])  # 小于基准值的放入right列表
    # 之后分别就是递归，分别对left,right列表实行快速排序
    return quick_sort(left) + [nums[0]] + quick_sort(right)  # 注意nums[0]这个元素要放在一个列表中


# 3、插入排序：将某一个数字插入到已经排好序的数组当中
def insert_sort(nums):
    n = len(nums)
    for i in range(1, n):  # 75234     # 1     # 2              #3             #4
        index = i                    #57234   #52734，25734 #25374，23574  #23547，23457
        for j in range(i - 1, -1, -1):  # 0   # 10             #210           #3210
            if nums[j] > nums[index]:
                # 第2个数，第3个数。。。每个数都要与其前面的数进行比较，如果比前面的数小，那么就要插入到该数之前。
                # 每次都是j=index-1。如果nums[j] > nums[index]成立，那么就将nums[j]与nums[index]交换位置，并将index-1。其实就相当于将nums[index]插到nums[j]的前面。
                nums[index], nums[j] = nums[j], nums[index]
                index -= 1
            else:
                break
    return nums


# 4、希尔排序
def shell_sort(nums):
    n = len(nums)
    gap = n//2   # 分组，在每组中，各自进行插入排序，随着gap的不断缩小，整体数据渐趋有序，插入排序的效率也越来越高。当最后gap=1时，就是普通的插入排序，但是因为数据已经大致有序所以效率很高。0,0+gap,0+gap+gap.....|1,1+gap,1+gap+gap....|2,....
    while gap > 0:
        for i in range(gap, n):
            while i > 0:
                if nums[i] < nums[i - gap]:
                    nums[i], nums[i - gap] = nums[i - gap], nums[i]
                    i = i - gap
                else:
                    break
        gap //= 2
    return nums


# 5、简单选择排序：每一次选择一个当前最小的元素放在已经排好序的数组后面
def head_sort(nums):
    n = len(nums)
    for i in range(n): # 从第一个数开始遍历          #75234      #0        #1             #2          #3       #4
        index = i                           #25734        23754         23457    2457
        for j in range(i + 1, n):  # 找出当前遍历到的第i+1个数及其后面的数中最小的数   #1234                         #234             #34             #4
            if nums[j] < nums[index]:
                index = j
        # 跳出循环之后，index中记录着当前遍历到的第i+1个数及其之后的数中最小的数的下标
        # 将找到的最小的数放到当前遍历到的第i+1个数的位置。也即放在之前排好序的数的后面
        nums[i], nums[index] = nums[index], nums[i]
    return nums


# 6、堆排序：构造最大堆，把最上面那个节点一道最后一个节点去
def sift_down(nums, start, end):
    root = start
    while True:
        # 从root开始对最大堆调整

        # 1、找左孩子节点
        child = 2 * root + 1
        if child > end:
            break

        # 2、找出两个child中较大的一个，最后只需把较大的child与root比较即可
        if child + 1 <= end and nums[child] < nums[child + 1]:
            child += 1

        # 3、root与child的值进行比较，保证root更大
        if nums[root] < nums[child]:
            nums[root], nums[child] = nums[child], nums[root]
            # ------调整过后，也要保证下面的堆仍然满足大根堆
            # 即把刚刚更改过的child作为节点，等着上面的步骤 4、再循环，保证child下面的对也是大根堆
            root = child
        else:
            # 没有调整，推出
            break


def heap_sort(nums):
    # 从最下层的最右端的根节点开始调整大根堆,然后是倒数第二个根节点，倒数第三个。。
    first = len(nums) // 2 - 1
    for start in range(first, -1, -1):
        sift_down(nums, start, len(nums) - 1)

    # 将最大的放到堆最后一个，堆-1，继续调整排序
    for end in range(len(nums) - 1, 0, -1):
        nums[0], nums[end] = nums[end], nums[0]
        sift_down(nums, 0, end - 1)  # 每次把最顶端的根节点换下去之后 start就是0。由于找到一个最大值把他移到最后面，就只要对剩下的继续找出最大值，所以所有节点数-1，即end-1

    return nums


# 7、计数排序
def counting_sort(nums):
    # 检查入参类型
    if not isinstance(nums, (list)):
        raise TypeError('error para type')
    # 获取nums中的最大值和最小值
    maxNum = max(nums)
    minNum = min(nums)
    # 以最大值和最小值的差作为中间数组的长度，并构建中间数组，初始化为0
    length = maxNum - minNum + 1
    tempArr = [0 for i in range(length)]
    # 创建结果List，从存放排序完成的结果
    resArr = list(range(len(nums)))
    # 第一次循环遍历 :用来遍历nums的每个元素，统计每个元素的出现次数，存入中间数组
    for num in nums:
        tempArr[num - minNum] += 1
    # 第二次循环 :遍历中间数组，每个位置的值=当前值+前一个位置的值，用来统计出小于等于当前下标的元素个数
    for j in range(1, length):
        tempArr[j] = tempArr[j] + tempArr[j - 1]
    # 第三次循环 :反向遍历nums的每个元素，找到该元素值在中间数组的对应下标，以这个中间数组的值作为结果数组的下标，将该元素存入结果数组
    for i in range(len(nums) - 1, -1, -1):
        resArr[tempArr[nums[i] - minNum] - 1] = nums[i]
        tempArr[nums[i] - minNum] -= 1
    return resArr


def main():
    nums = [4, 5, 3, 7, 2, 6, 8, 1, 9, 0]
    print(bubble_sort(nums))
    print(quick_sort(nums))
    print(insert_sort(nums))
    print(shell_sort(nums))
    print(head_sort(nums))
    print(heap_sort(nums))
    print(counting_sort(nums))


if __name__ == "__main__":
    main()
