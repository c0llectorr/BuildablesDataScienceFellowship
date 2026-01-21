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

def reverse_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev

def list_to_string(head):
    parts = []
    while head:
        parts.append(str(head.val))
        head = head.next
    return " -> ".join(parts)

def main():
    head = build_list([1, 2, 3, 4, 5])
    rev = reverse_list(head)
    print(list_to_string(rev))  

if __name__ == "__main__":
    main()
