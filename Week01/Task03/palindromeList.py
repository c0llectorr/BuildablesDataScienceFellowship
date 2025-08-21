class Node:
    def __init__(self, val, nxt=None):
        self.val = val
        self.next = nxt

def build_list(values):
    head = None
    tail = None
    for v in values:
        node = Node(v)
        if head is None:
            head = tail = node
        else:
            tail.next = node
            tail = node
    return head

def is_palindrome(head):
    if head is None or head.next is None:
        return True

    # Find middle (slow), end (fast)
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    prev = None
    curr = slow
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt

    # Compare halves
    first, second = head, prev
    while second:
        if first.val != second.val:
            return False
        first = first.next
        second = second.next
    return True

def main():
    word = input("Enter the first word: ")
    a = build_list(word)    
    print("The linked list is a palindrome." if is_palindrome(a) else "The linked list is not a palindrome.")
    
if __name__ == "__main__":
    main()
