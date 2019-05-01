
import pickle


def save(myobject, path):

    with open(path, "wb") as f:
        pickle.dump(myobject, f)


def load(path):

    with open(path, "rb") as f:
        myobject = pickle.load(f)

    return myobject