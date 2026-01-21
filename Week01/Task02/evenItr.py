class EvenNumbers:
    def __init__(self, limit):
        try:
            if limit < 0:
                raise ValueError("Limit must be non-negative")
            self.limit = limit
            self.current = 0
        except Exception as e:
            print(f"Initialization Error: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.current > self.limit:
                raise StopIteration
            num = self.current
            self.current += 2
            return num
        except StopIteration:
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise StopIteration
        
def main():
    try:
        limit = int(input("Enter the limit for even numbers: "))
        even_numbers = EvenNumbers(limit)
        print("Even numbers up to the limit:")
        for number in even_numbers:
            print(number, end=' ')
        print()
    except ValueError:
        print("Please enter a valid integer for the limit.")
        
if __name__ == "__main__":
    main()