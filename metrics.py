from keras.callbacks import Callback
from sklearn.metrics import make_scorer, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class MetricsCallback(Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.x = X_train
        self.y = y_train
        self.x_val = X_val
        self.y_val = y_val
        self.best_f1 = -float('Inf')
        self.best_prec = -float('Inf')
        self.best_recall = -float('Inf')
        self.best_roc_val = -float('Inf')
        self.acc = -float('Inf')
        self.epoch = 0
        super(MetricsCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        y_pred_val = self.model.predict_classes(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        f1_val = f1_score(self.y_val, y_pred_val)
        prec_val = precision_score(self.y_val, y_pred_val)
        recall_val = recall_score(self.y_val, y_pred_val)
        acc_val = accuracy_score(self.y_val, y_pred_val)
        if roc_val > self.best_roc_val:
            self.best_roc_val = roc_val
            self.best_f1 = f1_val
            self.best_prec = prec_val
            self.best_recall = recall_val
            self.best_acc = acc_val
            self.best_epoch = epoch + 1

        print('Best epoch: %s' % (str(self.best_epoch)))
        print('           acc val: %s  --        Best acc val: %s' % ("{:.4f}".format(acc_val),"{:.4f}".format(self.best_acc)))
        print('     precision val: %s  --  Best precision val: %s' % ("{:.4f}".format(prec_val), "{:.4f}".format(self.best_prec)))
        print('        recall val: %s  --     Best recall val: %s' % ("{:.4f}".format(recall_val), "{:.4f}".format(self.best_recall)))
        print('            f1 val: %s  --         Best f1 val: %s' % ("{:.4f}".format(f1_val), "{:.4f}".format(self.best_f1)))
        print('       ROC AUC val: %s  --    Best ROC AUC val: %s' % ("{:.4f}".format(roc_val), "{:.4f}".format(self.best_roc_val)))
        print('\n')
        
        return