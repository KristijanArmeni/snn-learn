
import pickle


def save(myobject, path):

    with open(path, "wb") as f:
        pickle.dump(myobject, f)


def load(path):

    with open(path, "rb") as f:
        myobject = pickle.load(f)

    return myobject


class Paths(object):

    def __init__(self):

        self.home = "/project/3011085.04/snn"
        self.raw = self.home + "/data/raw"
        self.interim = self.home + "/data/interim"
        self.results = self.home + "/data/figures"
