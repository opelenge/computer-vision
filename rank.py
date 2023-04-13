import sys

data = sys.stdin.readline().strip().split(' ')
data_sum= int(sys.stdin.readline())


new_data = []
for i in data:
  new_data.append(int(i))
def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    count = {}
    # print(nums)
    for i, num in enumerate(nums):
        remainder = target - num
        if remainder in count:
            return [count[remainder], i]
        count[num] = i
    return []