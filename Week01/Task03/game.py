class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

def build_trie(words):
    root = TrieNode()
    for w in words:
        node = root
        for ch in w:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.word = w
    return root

def find_words(board, words):
    if not board or not board[0]:
        return []

    root = build_trie(words)
    rows, cols = len(board), len(board[0])
    found = set()
    visited = [[False]*cols for _ in range(rows)]

    def dfs(r, c, node):
        ch = board[r][c]
        if ch not in node.children:
            return
        nxt = node.children[ch]
        if nxt.word:
            found.add(nxt.word)

        visited[r][c] = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    dfs(nr, nc, nxt)
        visited[r][c] = False

    for i in range(rows):
        for j in range(cols):
            dfs(i, j, root)

    return sorted(found)

def main():
    board = [
        ['G', 'I', 'Z'],
        ['U', 'E', 'K'],
        ['Q', 'S', 'E']
    ]
    dictionary = ["GEEKS", "QUIZ", "SEEK"]
    print(find_words(board, dictionary))

if __name__ == "__main__":
    main()
