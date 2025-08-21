def longest_pal_substring(s):
    if not s:
        return ""
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]

    best = s[0]
    for i in range(len(s)):
        p1 = expand(i, i)
        if len(p1) > len(best): best = p1
        p2 = expand(i, i + 1)
        if len(p2) > len(best): best = p2
    return best

def main():
    word = input("Enter a word to find the longest palindromic substring: ")
    print("Longest palindromic substring is:", longest_pal_substring(word))

if __name__ == "__main__":
    main()
