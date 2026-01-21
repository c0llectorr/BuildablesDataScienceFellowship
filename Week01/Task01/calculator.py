def calculator(number1, number2, operation):
    if operation == '+':
        return number1 + number2
    elif operation == '-':
        return number1 - number2
    elif operation == '*':
        return number1 * number2
    elif operation == '/':
        if number2 != 0:
            return number1 / number2
        else:
            raise ValueError("Cannot divide by zero.")
    else:
        raise ValueError("Invalid operation. Choose from: add, subtract, multiply, divide.")
    
def main():
    try:
        number1 = float(input("Enter first number: "))
        number2 = float(input("Enter second number: "))
        operation = input("Enter operation (+, -, *, /): ").strip()
        
        result = calculator(number1, number2, operation)
        print(f"The result is: {result}")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
        return
    
if __name__ == "__main__":
    main()