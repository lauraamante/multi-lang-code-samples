#Método de ordenação Quick Sort
#arr - array

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[-1]
    smaller = [x for x in arr[:-1] if x < pivot]
    equal   = [x for x in arr[:-1] if x == pivot]
    greater = [x for x in arr[:-1] if x > pivot]

    return quick_sort(smaller) + [pivot] + equal + quick_sort(greater)


if __name__ == "__main__":
    array = [7, 3, 6, 9, 8, 10]
    print(quick_sort(array))
