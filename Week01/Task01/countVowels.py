def countVowels(text):
    vowels = 'aeiouAEIOU'
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

def main():
    text = input("Enter a string: ")
    print(f"The number of vowels is: {countVowels(text)}")
    
if __name__ == "__main__":
    main()