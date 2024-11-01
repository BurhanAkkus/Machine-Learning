from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, x_train, y_train, x_test, y_test, learning_rate=0.01, epoch_limit=100, time_limit=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.epoch_limit = epoch_limit
        self.time_limit = time_limit

    @abstractmethod
    def train(self):
        """
        Modelin eğitim sürecini başlatır.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Girdi olarak verilen veri için modelin tahmin yapmasını sağlar.
        """
        pass

    @abstractmethod
    def test(self):
        """
        Test verisi üzerinde modeli değerlendirir.
        """
        pass

    @abstractmethod
    def summarize(self):
        """
        Modelin genel performansını ve öğrenme süreci hakkında özet bilgileri döndürür.
        """
        pass
