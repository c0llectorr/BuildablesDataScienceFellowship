def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def main():
    data = [(1, 3), (2, 6), (8, 10), (15, 18)]
    print(merge_intervals(data)) 

if __name__ == "__main__":
    main()
