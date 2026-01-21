def checkPalindrome(text):
    if text == text[::-1]:
        return True
    return False
def main():
    try:
        text = input("Enter a string: ")
        if checkPalindrome(text):
            print(f"{text} is a palindrome.")
        else:
            print(f"{text} is not a palindrome.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
if __name__ == "__main__":  
    main()