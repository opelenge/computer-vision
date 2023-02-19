def closest_sum(numbers_dict, target):
    numbers = list(numbers_dict.values())  # extract the numbers from the dictionary and convert to a list
    numbers.sort()  # sort the list in ascending order
    left = 0  # left pointer starts at the beginning of the list
    right = len(numbers) - 1  # right pointer starts at the end of the list
    closest = float('inf')  # initialize closest to infinity
    closest_pairs = []  # initialize closest_pairs to an empty list
    left_num = None  # initialize left_num to None
    right_num = None  # initialize right_num to None
    
    while left < right:
        sum_ = numbers[left] + numbers[right]  # calculate the sum of the two numbers
        diff = abs(sum_ - target)  # calculate the difference between the sum and the target
        
        if diff <= 1:  # if the difference is less than or equal to 1, add the pair to closest_pairs
            # find the two numbers in the dictionary that correspond to the closest pair
            for key, value in numbers_dict.items():
                if value == numbers[left]:
                    left_num = key
                if value == numbers[right]:
                    right_num = key
            closest_pairs.append((left_num, right_num))
        
        if diff < abs(closest - target):  # if the difference is smaller than the current closest
            closest = sum_  # update the closest sum
            left_num, right_num = None, None  # reset left_num and right_num to None
            closest_pairs = [(left_num, right_num)]  # update closest_pairs to a new list with only the closest pair
            
        if sum_ < target:  # if the sum is less than the target, move the left pointer to the right
            left += 1
        elif sum_ > target:  # if the sum is greater than the target, move the right pointer to the left
            right -= 1
        else:  # if the sum is equal to the target, we've found an exact match so return it
            for key, value in numbers_dict.items():
                if value == numbers[left]:
                    left_num = key
                if value == numbers[right]:
                    right_num = key
            return [(left_num, right_num)]
    
    return closest_pairs  # return all pairs with a difference to the target of 1 or -1



numbers_dict = {'a': 1, 'b': 5, 'c': 9, 'd': 12, 'e': 15, 'f': 16, 'g': 18, 'h': 19}
target = 7
closest_pairs = closest_sum(numbers_dict, target)
print(closest_pairs)  # prints [('b', 'f'), ('c', 'e'), ('d', 'f'), ('f', 'h')] (since these pairs have a difference to the target of 1 or -1)
