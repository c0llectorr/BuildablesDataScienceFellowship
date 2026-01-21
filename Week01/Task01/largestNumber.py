def findLargestNumber(number1, number2, number3):
    return max(number1, number2, number3)

def main():
    try:
        number1 = float(input("Enter first number: "))
        number2 = float(input("Enter second number: "))
        number3 = float(input("Enter third number: "))
        
        print(f"The largest number is: {findLargestNumber(number1, number2, number3)}")
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

if __name__ == "__main__":
    main()