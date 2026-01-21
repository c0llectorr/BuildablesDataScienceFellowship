def isPrime(number):
    if number <= 1:
        return False
    elif number == 2:
        return True
    else:
        for i in range(2, number-1):
            if (number % i) == 0:
                return False
    return True

def main():
    try:
        number = int(input("Enter a positive integer: "))
        if isPrime(number):
            print(f"{number} is a prime number.")
        else:
            print(f"{number} is not a prime number.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        
if __name__ == "__main__":  
    main()