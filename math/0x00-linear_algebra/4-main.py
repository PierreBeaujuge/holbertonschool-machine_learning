#!/usr/bin/env python3

add_arrays = __import__('4-line_up').add_arrays

arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
arr3 = add_arrays(arr1, arr2)
print(add_arrays(arr1, arr2))
print(arr1)
print(arr2)
print(add_arrays(arr1, [1, 2, 3]))
arr1.append(5)
arr2.append(9)
print(arr3)
print(arr1)
print(arr2)
arr5 = [[1, 2], [3, 4]]
arr6 = [[6, 7], [8, 9]]
arr7 = add_arrays(arr5, arr6)
print(add_arrays(arr5, arr6))
print(arr5)
print(arr6)
arr5[1].append(5)
arr6[1].append(10)
print(arr7)
print(arr5)
print(arr6)
