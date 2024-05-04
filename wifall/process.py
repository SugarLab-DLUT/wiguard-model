import csiread


def extract(file):
    csidata = csiread.Intel(file)
    csidata.read()
