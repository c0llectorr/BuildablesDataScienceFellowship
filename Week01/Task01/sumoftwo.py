def sumOfTwo (number1, number2):
    return number1 + number2

def main(): 
    try:
        number1 = float(input("Enter first number: "))
        number2 = float(input("Enter second number: "))
        result = sumOfTwo(number1, number2)
        print(f"The sum is: {result}")
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return
    
if __name__ == "__main__":
    main()