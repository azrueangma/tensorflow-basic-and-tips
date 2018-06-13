#-*- coding: utf-8 -*-
import tensorflow as tf
import os
import shutil
from utils import lazy_property
from layers import linear
from layers import relu_layer


class Model(object):
    def __init__(self, g, seed):
        pass


    def _build_model(self):
        pass


    @lazy_property
    def loss(self):
        pass


    @lazy_property
    def predict(self):
        pass


    @lazy_property
    def accuracy(self):
        pass


    @lazy_property
    def optim(self):
        pass


    @lazy_property
    def writer(self):
        pass


    @lazy_property
    def saver(self):
        pass


    @lazy_property
    def merged(self):
        pass


    @lazy_property
    def train_queue(self):
        pass


    @lazy_property
    def validation_queue(self):
        pass


    @lazy_property
    def test_queue(self):
        pass


    @lazy_property
    def input(self):
        pass


    def fit(self, model_reuse=False):
        pass


    def evaluation(self, x_test, y_test):
        pass


    def prediction(self, x_test):
        pass