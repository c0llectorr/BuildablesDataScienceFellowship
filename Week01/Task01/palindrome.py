def checkPalindrome(text):
    if text == text[::-1]:
        return True
    return False

def main():
    text = input("Enter a string: ").lower().replace(" ", "")
    if checkPalindrome(text):
        print(f"{text} is a palindrome.")
    else:
        print(f"{text} is not a palindrome.")
        
if __name__ == "__main__":
    main()