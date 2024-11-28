import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


train_data = pd.read_csv('digits.csv', header=None)
train_data.columns=["[0;0]","[0;1]","[0;2]","[0;3]","[0;4]","[0;5]","[0;6]","[0;7]",
                "[1;0]","[1;1]","[1;2]","[1;3]","[1;4]","[1;5]","[1;6]","[1;7]",
                "[2;0]","[2;1]","[2;2]","[2;3]","[2;4]","[2;5]","[2;6]","[2;7]",
                "[3;0]","[3;1]","[3;2]","[3;3]","[3;4]","[3;5]","[3;6]","[3;7]",
                "[4;0]","[4;1]","[4;2]","[4;3]","[4;4]","[4;5]","[4;6]","[4;7]",
                "[5;0]","[5;1]","[5;2]","[5;3]","[5;4]","[5;5]","[5;6]","[5;7]",
                "[6;0]","[6;1]","[6;2]","[6;3]","[6;4]","[6;5]","[6;6]","[6;7]",
                "[7;0]","[7;1]","[7;2]","[7;3]","[7;4]","[7;5]","[7;6]","[7;7]",
                "classe"]

features = train_data.columns[:-1]
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(train_data[features].values)
print(X_embedded.shape)


# séparation des données
digits_train = train_data.sample(frac=0.8, random_state=42)
digits_valid = train_data.drop(digits_train.index)

# X_train : les images matricielles d'entraînement
# Y_train : les chiffres représentés pas les images
X_train = digits_train[digits_train.columns[:-1]]
Y_train = digits_train[digits_train.columns[-1]]

# même chose sur les données de validation
X_valid = digits_valid[digits_train.columns[:-1]]
Y_valid = digits_valid[digits_train.columns[-1]]


# la distance
def euclidean_distance(v1, v2):
    return math.sqrt(sum((b - a) ** 2 for a, b in zip(v1, v2)))


# tests sur les distances
# print("classe de la 1ère ligne : ", Y_train.iloc[0])
# print("classe de la 2ème ligne : ", Y_train.iloc[1])
# print("classe de la 26ème ligne : ", Y_train.iloc[25])

# print("Distance entre la ligne 0 et la ligne 1 : ", euclidean_distance(X_train.iloc[0],X_train.iloc[1]))
# print("Distance entre la ligne 0 et la ligne 25 : ", euclidean_distance(X_train.iloc[0],X_train.iloc[25]))


# les k plus proches voisins
def neighbors(X_train, y_label, x_test, k):
    list_distances = []

    # Calculer la liste des distances par rapport à la donnée x_test
    for i in range(X_train.shape[0]):
        list_distances.append(euclidean_distance(X_train.iloc[i], x_test))

    df = pd.DataFrame()

    df["label"] = y_label
    df["distance"] = list_distances

    df = df.sort_values(by="distance")

    return df.iloc[:k,:]


# la prédiction sur la liste des k voisins trouvés
def prediction(neighbors):
    tab_effectifs = [0] * 10
    for chiffre in neighbors['label']:
        tab_effectifs[chiffre] += 1
    return tab_effectifs.index(max(tab_effectifs))


test0 = [0, 0, 3, 13, 14, 2, 0, 0, 0, 2, 15, 14, 13, 15, 4, 0, 0, 11, 16, 4, 6, 12, 15, 1, 1, 16, 12, 0, 0, 6, 16, 4, 4, 16, 8, 0, 0, 6, 16, 6, 4, 16, 10, 0, 0, 11, 16, 3, 0, 13, 15, 8, 10, 16, 12, 0, 0, 2, 12, 16, 16, 12, 2, 0]

nearest_neighbors = neighbors(X_train, Y_train, test0, 5)
print("prédiction sur un 0 : ", end= "")
print(prediction(nearest_neighbors))


# fonction qui évalue la prédiction par rapport à la réalité
# sur les données de validation
def evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=True):

    TP=0 # vrai
    FP=0 # faux

    total = 0

    for i in range(X_valid.shape[0]):

        nearest_neighbors = neighbors(X_train, Y_train, X_valid.iloc[i], k)

        if (prediction(nearest_neighbors) == Y_valid.iloc[i]):
            TP += 1
        elif ((prediction(nearest_neighbors) != Y_valid.iloc[i])):
            FP+=1
            print("Erreur de prédiction : ")
            print("Prediction : ", prediction(nearest_neighbors))
            print("Résultat attendu : ", Y_valid.iloc[i])
        total += 1 

    accuracy = (TP)/total

    if verbose:
        print("Accuracy:" + str(accuracy))

    return accuracy


# recherche du meilleur paramètre k
# list_accuracy = []
# for k in range(1,11,2):
#     list_accuracy.append(evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=False))
# print(list_accuracy)

# les tests avec des données non étiquetées
test0 = [0, 0, 3, 13, 14, 2, 0, 0, 0, 2, 15, 14, 13, 15, 4, 0, 0, 11, 16, 4, 6, 12, 15, 1, 1, 16, 12, 0, 0, 6, 16, 4, 4, 16, 8, 0, 0, 6, 16, 6, 4, 16, 10, 0, 0, 11, 16, 3, 0, 13, 15, 8, 10, 16, 12, 0, 0, 2, 12, 16, 16, 12, 2, 0]
test1 = [0, 0, 0, 0, 0, 11, 11, 0, 0, 0, 0, 0, 8, 16, 12, 0, 0, 1, 5, 12, 16, 16, 12, 0, 0, 8, 16, 15, 9, 16, 11, 0, 0, 0, 2, 1, 2, 16, 10, 0, 0, 0, 0, 0, 2, 16, 9, 0, 0, 0, 0, 0, 1, 16, 8, 0, 0, 0, 0, 0, 0, 13, 6, 0]
test2 = [0, 2, 12, 15, 12, 1, 0, 0, 0, 6, 14, 11, 16, 4, 0, 0, 0, 1, 1, 11, 15, 1, 0, 0, 0, 0, 3, 16, 8, 0, 0, 0, 0, 0, 12, 15, 1, 0, 0, 0, 0, 3, 16, 8, 0, 0, 0, 0, 0, 7, 16, 6, 4, 8, 4, 0, 0, 2, 14, 16, 15, 10, 2, 0]
test3 = [0, 9, 16, 14, 6, 0, 0, 0, 0, 6, 11, 15, 16, 6, 0, 0, 0, 0, 4, 14, 16, 5, 0, 0, 0, 2, 15, 16, 16, 9, 1, 0, 0, 0, 2, 5, 11, 16, 10, 0, 0, 0, 0, 0, 5, 16, 15, 0, 0, 2, 6, 8, 15, 16, 9, 0, 0, 7, 16, 16, 14, 8, 0, 0]
test4 = [0, 0, 0, 9, 15, 1, 0, 0, 0, 0, 5, 16, 7, 0, 0, 0, 0, 0, 13, 13, 0, 0, 0, 0, 0, 2, 16, 6, 0, 4, 0, 0, 0, 1, 13, 12, 13, 16, 2, 0, 0, 0, 2, 8, 14, 14, 0, 0, 0, 0, 0, 0, 12, 11, 0, 0, 0, 0, 0, 0, 9, 9, 0, 0]
test5 = [0, 2, 14, 13, 8, 1, 0, 0, 0, 5, 16, 10, 9, 5, 0, 0, 0, 7, 16, 4, 0, 0, 0, 0, 0, 8, 16, 4, 0, 0, 0, 0, 0, 8, 16, 16, 12, 2, 0, 0, 0, 1, 7, 7, 15, 15, 2, 0, 0, 0, 7, 5, 5, 16, 10, 0, 0, 1, 11, 15, 16, 16, 9, 0]
test6 = [0, 0, 0, 0, 8, 15, 7, 0, 0, 0, 0, 7, 16, 14, 3, 0, 0, 0, 5, 16, 14, 1, 0, 0, 0, 1, 14, 15, 3, 0, 0, 0, 0, 6, 16, 13, 7, 1, 0, 0, 0, 7, 16, 12, 8, 14, 1, 0, 0, 4, 16, 10, 10, 16, 2, 0, 0, 0, 7, 15, 16, 10, 0, 0]
test7 = [0, 3, 11, 12, 14, 15, 6, 0, 0, 3, 13, 15, 16, 16, 6, 0, 0, 0, 0, 0, 15, 16, 2, 0, 0, 0, 0, 3, 16, 13, 0, 0, 0, 6, 12, 13, 16, 14, 8, 0, 0, 6, 15, 16, 16, 15, 12, 0, 0, 0, 0, 12, 16, 4, 0, 0, 0, 0, 0, 11, 16, 2, 0, 0]
test8 = [0, 0, 0, 12, 15, 11, 8, 0, 0, 0, 2, 16, 10, 6, 4, 1, 0, 0, 1, 16, 9, 0, 1, 5, 0, 0, 0, 15, 11, 0, 9, 8, 0, 0, 0, 12, 15, 14, 15, 2, 0, 3, 11, 16, 16, 9, 2, 0, 3, 15, 16, 16, 14, 0, 0, 0, 0, 9, 15, 16, 8, 0, 0, 0]
test9 = [0, 0, 5, 13, 15, 5, 0, 0, 0, 3, 16, 9, 10, 3, 0, 0, 0, 6, 16, 2, 7, 11, 2, 0, 0, 1, 10, 16, 16, 16, 5, 0, 0, 0, 0, 2, 15, 9, 0, 0, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0, 5, 4, 4, 16, 4, 0, 0, 1, 10, 13, 16, 16, 6, 0]

print("TESTS")
mes_tests = [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9]

for i in range(len(mes_tests)):
    mon_chiffre = mes_tests[i]
    nearest_neighbors = neighbors(X_train, Y_train, mon_chiffre, 5)
    la_prediction = prediction(nearest_neighbors)
    if i != la_prediction:
        print(f"Mauvaise prédiction sur {i} : {la_prediction}")
        plt.imshow(np.array(mon_chiffre).reshape(8, 8), cmap='Greys')
        plt.show()
