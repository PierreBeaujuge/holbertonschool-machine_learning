#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))

poly = [5, '3', 0, 1]
print(poly_integral(poly))

poly = []
print(poly_integral(poly))

poly = [5, 3, 0.5, 1]
print(poly_integral(poly))

C = 1
print(poly_integral(poly, C))

C = '1'
print(poly_integral(poly, C))

C = 1.5
print(poly_integral(poly, C))
