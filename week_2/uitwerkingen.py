import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# ==== OPGAVE 1 ====
def plotNumber(nrVector):
    # Let op: de manier waarop de data is opgesteld vereist dat je gebruik maakt
    # van de Fortran index-volgorde – de eerste index verandert het snelst, de 
    # laatste index het langzaamst; als je dat niet doet, wordt het plaatje 
    # gespiegeld en geroteerd. Zie de documentatie op 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
    
    plt.matshow(nrVector.reshape((20, 20), order='F'))
    plt.show()

# ==== OPGAVE 2a ====
def sigmoid(z):
    # Maak de code die de sigmoid van de input z teruggeeft. Zorg er hierbij
    # voor dat de code zowel werkt wanneer z een getal is als wanneer z een
    # vector is.
    # Maak gebruik van de methode exp() in NumPy.

    return 1 / (1 + np.exp(-z))


# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    # Gegeven een vector met waarden y_i van 1...x, retourneer een (ijle) matrix
    # van m×x met een 1 op positie y_i en een 0 op de overige posities.
    # Let op: de gegeven vector y is 1-based en de gevraagde matrix is 0-based,
    # dus als y_i=1, dan moet regel i in de matrix [1,0,0, ... 0] zijn, als
    # y_i=10, dan is regel i in de matrix [0,0,...1] (in dit geval is de breedte
    # van de matrix 10 (0-9), maar de methode moet werken voor elke waarde van 
    # y en m
    row = np.arange(m)
    col = np.subtract(y, 1).reshape(m)
    return csr_matrix((np.ones(m), (row, col))).todense()


# ==== OPGAVE 2c ==== 
# ===== deel 1: =====
def predictNumber(Theta1, Theta2, X):
    # Deze methode moet een matrix teruggeven met de output van het netwerk
    # gegeven de waarden van Theta1 en Theta2. Elke regel in deze matrix 
    # is de waarschijnlijkheid dat het sample op die positie (i) het getal
    # is dat met de kolom correspondeert.

    # De matrices Theta1 en Theta2 corresponderen met het gewicht tussen de
    # input-laag en de verborgen laag, en tussen de verborgen laag en de
    # output-laag, respectievelijk. 

    # Een mogelijk stappenplan kan zijn:

    #    1. voeg enen toe aan de gegeven matrix X; dit is de input-matrix a1
    #    2. roep de sigmoid-functie van hierboven aan met a1 als actuele
    #       parameter: dit is de variabele a2
    #    3. voeg enen toe aan de matrix a2, dit is de input voor de laatste
    #       laag in het netwerk
    #    4. roep de sigmoid-functie aan op deze a2; dit is het uiteindelijke
    #       resultaat: de output van het netwerk aan de buitenste laag.

    # Voeg enen toe aan het begin van elke stap en reshape de uiteindelijke
    # vector zodat deze dezelfde dimensionaliteit heeft als y in de exercise.

    m, n = X.shape
    a1 = np.c_[np.ones(m), X]
    a2 = sigmoid(np.dot(a1, Theta1.T))
    
    m, n = a2.shape
    a2 = np.c_[np.ones(m), a2]
    out = sigmoid(np.dot(a2, Theta2.T))

    return out



# ===== deel 2: =====
def computeCost(Theta1, Theta2, X, y):
    # Deze methode maakt gebruik van de methode predictNumber() die je hierboven hebt
    # geïmplementeerd. Hier wordt het voorspelde getal vergeleken met de werkelijk 
    # waarde (die in de parameter y is meegegeven) en wordt de totale kost van deze
    # voorspelling (dus met de huidige waarden van Theta1 en Theta2) berekend en
    # geretourneerd.
    # Let op: de y die hier binnenkomt is de m×1-vector met waarden van 1...10. 
    # Maak gebruik van de methode get_y_matrix() die je in opgave 2a hebt gemaakt
    # om deze om te zetten naar een matrix. 

    m, n = X.shape
    Y = get_y_matrix(y, m)
    P = predictNumber(Theta1, Theta2, X)
    J = np.multiply(-Y, np.log(P)) - np.multiply(1 - Y, np.log(1 - P))

    return J.sum(axis=1).mean()


# ==== OPGAVE 3a ====
def sigmoidGradient(z): 
    # Retourneer hier de waarde van de afgeleide van de sigmoïdefunctie.
    # Zie de opgave voor de exacte formule. Zorg ervoor dat deze werkt met
    # scalaire waarden en met vectoren.
    g = sigmoid(z)
    return np.multiply(g, 1 - g)

# ==== OPGAVE 3b ====
def nnCheckGradients(Theta1, Theta2, X, y): 
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape)
    m = 5000 #voorbeeldwaarde; dit moet je natuurlijk aanpassen naar de echte waarde van m

    ysparse = get_y_matrix(y, y.shape[0])
    for i in range(m):
        #YOUR CODE HERE
        m, n = X.shape
        a1 = np.concatenate((np.ones(1), X[i]), axis=0)
        z1 = np.dot(a1, Theta1.T)
        a2 = sigmoid(z1)


        a2 = np.concatenate((np.ones(1), a2), axis=0)
        z2 = np.dot(a2, Theta2.T)
        a3 = sigmoid(z2)

        d3 = a3 - ysparse[i]

        sigmoidgrad = np.concatenate((np.ones(1), sigmoidGradient(z1)), axis=0)
        d2 = np.multiply(np.dot(d3, Theta2), sigmoidgrad.T) # 1,26 * 1.10 moet 1.26*1.26 zijn

        d2 = np.delete(d2, 0)




        #Delta3 += np.dot(d3, a2.T) #gegeven in vragenuur
        Delta3 += np.dot(d3.T, a2.reshape(1, 26))
        #Delta2 += np.dot(d2, a1.T) #gegeven in vragenuur
        Delta2 += np.dot(d2.T, a1.reshape(1, 401))



    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m

    return Delta2_grad, Delta3_grad
