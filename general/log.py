from general.singleton import Singleton


class Log(Singleton):
    def __init__(self):
        log = None