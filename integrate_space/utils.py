import numpy as np
import os
import csv


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d

def make_directory(path):
    try:
        os.mkdir(path)
        print(path + ' is generated!')
    except OSError:
        pass

def _csv_writer(file_name, write_data):
    f = open(file_name, 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(write_data)
    f.close()

def logger(log_data):
    _csv_writer('log.csv', log_data)