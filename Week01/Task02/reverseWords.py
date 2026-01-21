def reverse_words(sentence):
    words = sentence.split()
    return " ".join(words[::-1])


def main():
    sentence = input("Enter a sentence: ")
    reversed_sentence = reverse_words(sentence)
    print(f"Reversed sentence: {reversed_sentence}")
    
if __name__ == "__main__":
    main()