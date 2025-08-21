def merge_sorted(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out

def main():
    arr1 = []
    arr2 = []
    while True:
        try:
            arr1 = list(map(int, input("Enter first sorted list : ").split()))
            arr2 = list(map(int, input("Enter second sorted list : ").split()))
            break
        except ValueError:
            print("Please enter valid integers.")
            
    arr1.sort()
    arr2.sort()
    print(merge_sorted(arr1, arr2)) 

if __name__ == "__main__":
    main()
