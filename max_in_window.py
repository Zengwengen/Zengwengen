def max_in_windows(nums, size_1):
    if not nums or size_1 <= 0 or size_1 > len(nums):
        return []
    deque, res = [], []
    for i in range(len(nums)):
        while deque and nums[i] > nums[deque[-1]]:
            deque.pop()
        deque.append(i)
        if deque[0] == i - size_1:
            deque.pop(0)
        if i >= size_1 - 1:
            res.append(nums[deque[0]])
    return res


if __name__ == '__main__':
    num = list(map(int, input("请输入数组,用空格符号分隔：").split()))
    size = int(input("请输入滑动窗口大小："))
    re = max_in_windows(num, size)
    print("所有滑动窗口的最大值为：", re)
