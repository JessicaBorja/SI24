test_cases = [{"input": [1, 1, 2, 2, 2], "expected_output": 2},
              {"input": [5, 5, 5, 5, 2, 2, 2], "expected_output": 5},
              {"input": [-1, -1, -1, 0, 0, 2, 2], "expected_output": -1}]
def elemento_mayoritario(nums):
    mayoritario = None
    contador = 0
    
    for num in nums:
        if contador == 0:
            mayoritario = num
            contador = 1
        elif num == mayoritario:
            contador += 1
        else:
            contador -= 1
    
    return mayoritario