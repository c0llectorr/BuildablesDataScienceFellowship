def evenOrOdd (number):
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"
    
def main():
    try:
        number = int(input("Enter a number: "))
        print(f"{number} is {evenOrOdd(number)}.")
        
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return
if __name__ == "__main__":
    main()