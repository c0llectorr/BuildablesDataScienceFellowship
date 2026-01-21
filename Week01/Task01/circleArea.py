import math

def calculateArea (radius):
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * radius * radius

def main():
    try:
        radius = float(input("Enter the radius of the circle: "))
        area = calculateArea(radius)
        print(f"The area of the circle is: {area}")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
        return
if __name__ == "__main__":
    main()