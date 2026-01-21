def isAnagram(word1, word2):
    if sorted(word1.lower().strip()) == sorted(word2.lower().strip()):
        return True
    return False

def main():
    word1 = input("Enter the first word: ")
    word2 = input("Enter the second word: ")
    
    if isAnagram(word1, word2):
        print(f'"{word1}" and "{word2}" are anagrams.')
    else:
        print(f'"{word1}" and "{word2}" are not anagrams.')
        
if __name__ == "__main__":
    main()