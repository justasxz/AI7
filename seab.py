import seaborn as sns
import matplotlib.pyplot as plt

# data = [-5,2,9,4,6]
# sns.set_theme(style="darkgrid")
# sns.lineplot(data)
# plt.title("Pavyzdinė linijinė diagrama")
# plt.xlabel("Indeksas")
# plt.ylabel("Reikšmė")

# plt.show()

# print("Programa baigta")
# data = [-5,2,9,4,6]
# indexes = [1,2,3,4,5]
# sns.set_theme(style="darkgrid")
# sns.lineplot(x=indexes, y=data)
# plt.title("Pavyzdinė linijinė diagrama")
# plt.xlabel("Indeksas")
# plt.ylabel("Reikšmė")

# plt.show()

data_for_hist = [7,7,8,9,9,9,1,1,1,1,1,2,2,3,3,4,5,6,6,6,7,8,9]
# sns.histplot(data_for_hist, bins=4)
# plt.show()
# print("Programa baigta")
sns.countplot(data=data_for_hist)
plt.show()