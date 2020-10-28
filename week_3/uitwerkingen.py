import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# OPGAVE 1a
def plotImage(img, label):
    # Deze methode krijgt een matrix mee (in img) en een label dat correspondeert met het 
    # plaatje dat in de matrix is weergegeven. Zorg ervoor dat dit grafisch wordt weergegeven.
    # Maak gebruik van plt.cm.binary voor de cmap-parameter van plt.imgshow.

    # YOUR CODE HERE
    plt.imshow(img, cmap='binary')
    plt.title(label=label)
    plt.show()



# OPGAVE 1b
def scaleData(X):
    # Deze methode krijgt een matrix mee waarin getallen zijn opgeslagen van 0..m, en hij 
    # moet dezelfde matrix retourneren met waarden van 0..1. Deze methode moet werken voor 
    # alle maximal waarde die in de matrix voorkomt.
    # Deel alle elementen in de matrix 'element wise' door de grootste waarde in deze matrix.

    # YOUR CODE HERE
    return np.divide(X, np.amax(X))

# OPGAVE 1c
def buildModel():
    # Deze methode maakt het keras-model dat we gebruiken voor de classificatie van de mnist
    # dataset. Je hoeft deze niet abstract te maken, dus je kunt er van uitgaan dat de input
    # layer van dit netwert alleen geschikt is voor de plaatjes in de opgave (wat is de 
    # dimensionaliteit hiervan?).
    # Maak een model met een input-laag, een volledig verbonden verborgen laag en een softmax
    # output-laag. Compileer het netwerk vervolgens met de gegevens die in opgave gegeven zijn
    # en retourneer het resultaat.

    model = None

    # YOUR CODE HERE
    model = keras.Sequential()
    #model.add(tf.keras.Input(shape=(28 * 28,)))
    #model.add(tf.keras.layers.Dense(784))
    #TODO de eerste layer moet 28*28 matrix veranderen in 784
    model.add(keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(28*28,)))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics="accuracy")


    # inputs = keras.layers.Input(input_shape=(28, 28))
    # x = keras.layers.Dense(128, activation=tf.nn.relu)(inputs)
    # outputs = keras.layers.Dense(10, activation=tf.nn.softmax)(x)
    # model = keras.Model(inputs=inputs, outputs=outputs)
    # model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
    return model


# OPGAVE 2a
def confMatrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix
    
    # YOUR CODE HERE
    cf = tf.math.confusion_matrix(labels, pred)
    return cf

# OPGAVE 2b
def confEls(conf, labels): 
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)
 
    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
 
    # YOUR CODE HERE
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    tn = conf.sum() - (tp + fp + fn)

    return list(zip(labels, tp, fp, fn, tn))

# OPGAVE 2c
def confData(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    # VERVANG ONDERSTAANDE REGELS MET JE EIGEN CODE
    #TODO werkt niet
    tp = sum([metric[1] for metric in metrics])
    fp = sum([metric[2] for metric in metrics])
    fn = sum([metric[3] for metric in metrics])
    tn = sum([metric[4] for metric in metrics])


    # BEREKEN HIERONDER DE JUISTE METRIEKEN EN RETOURNEER DIE 
    # ALS EEN DICTIONARY

    TPR = tp/(tp+fn)
    PPV = tp/(tp+fp)
    TNR = tn/(tn+fp)
    FPR = fp/(fp+tn)

    rv = {'tpr': TPR, 'ppv': PPV, 'tnr': TNR, 'fpr': FPR }
    return rv
