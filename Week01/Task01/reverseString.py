def reverseString(string):
    return string[::-1]

def main():
    try:
        string = input("Enter a string: ")
        reversed_string = reverseString(string)
        print(f"The reversed string is: {reversed_string}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return
if __name__ == "__main__":  
    main()