def feb_seq(n):
    if n <= 0:
        return 0
    elif n == 1:
        return [0, 1]
    elif n == 2:
        return [0, 1, 1]
    else:
        a, b = 1, 1
        seq = [0, 1, 1]
        for i in range(2, n-1):
            a, b = b, a + b
            seq.append(b)
        return seq
    
def main():
   try: 
       number = int(input("Enter a positive integer: "))
       print(f"Fibonacci sequence: {feb_seq(number)} ")
    
   except ValueError:
       print("Please enter a number.")
       
if __name__ == "__main__":
    main()