def max_subarray_sum(nums):
    curr = max_sum = nums[0]
    for x in nums[1:]:
        curr = max(x, curr + x)
        max_sum = max(max_sum, curr)
    return max_sum

def main():
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(max_subarray_sum(arr)) 

if __name__ == "__main__":
    main()
