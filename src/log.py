# coding: utf-8
import sys
import logging


def setup_logging(name, stdout=False):
    fmt = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # fmt = "%(message)s"
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        filename=name)
    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(logging.Formatter("%(message)s"))
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
    return logging.getLogger(name)


class Logger(object):
    def __init__(self, name="a.out", filemode='a'):
        self.terminal = sys.stdout
        self.log = open(name, filemode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        self.log.close()

if __name__ == '__main__':
    # logger = setup_logging("test_log.txt")
    # logger.info("good job")
    # INFO log.py:  22: good job

    sys.stdout = Logger()

    print("hello ss")
    print("hello 6666")
