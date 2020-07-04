import numpy as np

idx1 = np.array([0, 3, 1, 4, 2])
idx2 = np.array([1, 4, 0, 3, 2])
idx3 = np.array([0,  1,  2,  3, 6, 7,  4, 5])
idx4 = np.array([0,  1,  2,  3, 4, 6,  5, 7])
idx5 = np.array([0,  1,  2,  3,  4, 10,  5,  9,  6, 11,  7,  8])
idx6 = np.array([0,  1,  2,  3, 10, 11,  4,  5,  9,  6,  7,  8])


def test(arr1, arr2):
    temp = []
    start = 0

    for item in np.nditer(arr1):
        ok = (np.argwhere(arr2 == item)[0][0] >= start)
        if ok:
            temp.append(item)
        start += 1
    return np.array(temp)


new_idx = np.array([])
for i in range(len(idx1)):
    temp1 = idx1[i:]
    j = np.argwhere(idx2 == idx1[i])
    if j.size > 0:
        temp2 = idx2[j[0][0]:]
        ok = True  # test(temp1.tolist(), temp2.tolist())
        res = temp1[np.in1d(temp1, temp2)]
        if res.size > new_idx.size:
            new_idx = res
        # print(str(ok) + ': ' + str(temp1) + ' - ' + str(temp2) + ' = ' + str(res))

# print(new_idx)
# print(test(idx1, idx2))

t1 = np.empty((2, 5))
t1[0] = idx1
t1[1] = idx2
print(t1.T)
