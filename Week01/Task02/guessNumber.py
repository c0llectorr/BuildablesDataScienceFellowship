import random

key = random.randint(1, 100)


def guessNumber(number):
    if number > key:
        return "Too high!"
    elif number < key:
        return "Too low!"
    else:
        return "Correct!"
    
def main():
    print("Select a number between 1 and 100.")
    while True:
        try:
            guess = int(input("Please enter your guess: "))
            result = guessNumber(guess)
            print(result)
            if result == "Correct!":
                break
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 100.")
            
if __name__ == "__main__":
    main()