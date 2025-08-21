class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

    def insert(self, key):
        if key < self.key:
            if self.left: self.left.insert(key)
            else: self.left = BSTNode(key)
        elif key > self.key:
            if self.right: self.right.insert(key)
            else: self.right = BSTNode(key)

    def search(self, key):
        if key == self.key:
            return True
        if key < self.key and self.left:
            return self.left.search(key)
        if key > self.key and self.right:
            return self.right.search(key)
        return False

    def _min_node(self):
        node = self
        while node.left:
            node = node.left
        return node

    def delete(self, key):
        if key < self.key:
            if self.left:
                self.left = self.left.delete(key)
        elif key > self.key:
            if self.right:
                self.right = self.right.delete(key)
        else:
          
            if not self.left:  return self.right
            if not self.right: return self.left
         
            succ = self.right._min_node()
            self.key = succ.key
            self.right = self.right.delete(succ.key)
        return self

    def inorder_print(self):
        if self.left: self.left.inorder_print()
        print(self.key, end=" ")
        if self.right: self.right.inorder_print()

def main():
    # Build BST
    vals = [50, 30, 70, 20, 40, 60, 80]
    root = BSTNode(vals[0])
    for v in vals[1:]:
        root.insert(v)

    print("In-order traversal after inserts:")
    root.inorder_print(); print()

    print("Search 60:", root.search(60))
    print("Search 25:", root.search(25))

    # Deletions
    for d in [20, 30, 50]:
        root = root.delete(d)
        print(f"In-order after deleting {d}:")
        if root:
            root.inorder_print(); print()

if __name__ == "__main__":
    main()
