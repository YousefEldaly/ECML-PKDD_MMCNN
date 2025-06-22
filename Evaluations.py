import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score

class Evaluations():
    def __init__(self,
                 history,
                 y_pred,
                 y_true,
                 loss_score,
                 error,
                 validation_score):
        self.history = history
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss_score = loss_score
        self.error = error
        self.validation_score = validation_score
        
        self.matrix, self.kappa = self.matrix_and_kappa(self.y_pred, self.y_true)
        self.f1 = self.compute_f1_score(self.y_pred, self.y_true)  # Renamed to avoid conflict
        
    # Matrix confusion and the kappa value
    def matrix_and_kappa(self, y_pred, y_true):
        # Convert probabilities to class labels
        y_pred_labels = []
        for i in y_pred:
            if i[0] > i[1]:
                y_pred_labels.append(0)
            else:
                y_pred_labels.append(1)
        
        y_true_labels = []
        for i in y_true:
            if i[0] > i[1]:
                y_true_labels.append(0)
            else:
                y_true_labels.append(1)
        
        C2 = confusion_matrix(y_true_labels, y_pred_labels)
        kappa_value = cohen_kappa_score(y_true_labels, y_pred_labels)
        return C2, kappa_value
    
    # f1 score calculation (renamed to avoid conflict with sklearn's f1_score)
    def compute_f1_score(self, y_pred, y_true):
        # Convert probabilities to class labels
        y_pred_labels = []
        for i in y_pred:
            if i[0] > i[1]:
                y_pred_labels.append(0)
            else:
                y_pred_labels.append(1)
        
        y_true_labels = []
        for i in y_true:
            if i[0] > i[1]:
                y_true_labels.append(0)
            else:
                y_true_labels.append(1)
        
        f1 = f1_score(y_true_labels, y_pred_labels, average=None)
        return f1
    
    def draw_pict(self, history, types=1):
        if types == 1:
            plt.plot(history.history['binary_accuracy'])
            plt.plot(history.history['val_binary_accuracy'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='lower right')  
            plt.title('accuracy')
            plt.show()
            
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('loss') 
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='upper right')  
            plt.title('loss')
            plt.show()
        else:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='lower right')  
            plt.title('accuracy')
            plt.show()
            
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.ylabel('loss') 
            plt.xlabel('epoch')
            plt.legend(('train', 'validation'), loc='upper right')  
            plt.title('loss')
            plt.show()
