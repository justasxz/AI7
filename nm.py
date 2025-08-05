import numpy as np

# sarasas = [1,4,9,6,4]
# from_list = np.array(sarasas)
# print(from_list)
# print(np.random.randint(0,100,(10,5)))
# sarasas = [4]
# sarasas2 = [1,2,3,4,5]
# print(sarasas + sarasas2)

# narray = np.arange(1,51,0.5) # range(start, stop, step)
# print(narray)

# print(np.max(narray))
# narray = narray.reshape(25,4)
# print(narray)
# # add to narray
# # narray = np.append(narray, [["labas","Kapas","tapas","Melas"]], axis=0)
# # narray.sum()
# print(narray.dtype)

# narray = np.array([
#     [7,5,3,1], # 0
#     [2,4,6,8], # 1
#     [9,3,5,7], # 2
#     [1,2,3,4] # 3
# ])

# print(narray)

# print(narray[1:3, 1:3])

# print(narray[narray > 5])

# e - konstanta (dar vadinama Eulerio skaičiumi) e = 2.718

# npiarray = np.array([0, np.pi/2, np.pi])
# npiarray = np.array([0, 1, 2])
# print(np.exp(npiarray))  # e^x, kur x yra npiarray elementai

# narray = np.array([160,180,190,210,150,155,175,184,198,172,230,140]) # 4 5 8 9 10
# # print(np.median(narray))  # mediana
# print(np.mean(narray))    # vidurkis
# print(np.std(narray))     # standartinis nuokrypis
narray1 = np.array([1,2,3,4,5])
narray2 = np.array([6,7,8,9,10])

print(narray1 + 5)  # sudeti
print(narray1 - narray2)  # atimti
print(narray1 * narray2)  # dauginti
print(narray1 / narray2)  # dalinti