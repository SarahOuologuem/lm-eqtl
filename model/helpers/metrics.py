import torch
from torchmetrics import Metric


class MaskedAccuracy(Metric):
    """
    Computes the Accuracy for only the masked nucleotides.
    All target values holding the ignore index will be ignored during the accuracy computation
    """
        
    # Set to True if the metric is differentiable else set to False
    is_differentiable: bool = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, smooth=False, smooth_beta=0.98):
        super().__init__()
        #self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        #self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cumsum", default=torch.tensor(0.), dist_reduce_fx="sum")
    
        self.ignore_index = -100.0
        
        self.smooth_beta = smooth_beta
        self.smooth = smooth
        self.itr_idx = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        #self.correct += torch.sum(preds == target)
        #self.total += torch.sum(target != self.ignore_index) # ignore masked ones
        
        current_correct = torch.sum(preds == target)
        current_total = torch.sum(target != self.ignore_index) # ignore masked ones
        
        if current_total>0:
            current_acc = current_correct / current_total
            if self.smooth:
                self.cumsum = self.smooth_beta * self.cumsum + (1-self.smooth_beta)*current_acc
            else:
                self.cumsum += current_acc
            self.itr_idx += 1

    def compute(self):
        """
        Divide correct predictions by all predictions (ignoring the masked ones)
        """
        if self.smooth:
            return self.cumsum / (1 - self.smooth_beta**(self.itr_idx))
        else:
            return self.cumsum / self.itr_idx
        #if self.total != 0:
        #    return (self.correct.float() / self.total).item()
        #return 0 # if we gotta divide by 0

class MeanRecall(Metric):
    """
    Computes the everage recall of classes 0-3 for only the masked nucleotides.
    All target values holding the ignore index will be ignored during the accuracy computation
    """
        
    # Set to True if the metric is differentiable else set to False
    is_differentiable: bool = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, Nclasses=2):
        super().__init__()
        self.add_state("correct", default=torch.zeros(Nclasses), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(Nclasses), dist_reduce_fx="sum")
        self.ignore_index = -100.0
        self.Nclasses = Nclasses

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        for class_idx in range(self.Nclasses):
            true_pos = preds == target
            total_pos = target == class_idx
            self.correct[class_idx] += torch.sum(true_pos & total_pos)
            self.total[class_idx] += torch.sum(total_pos) # ignore masked ones

    def compute(self):
        """
        Divide correct predictions by all predictions (ignoring the masked ones)
        """
        return (self.correct.float() / self.total).detach().cpu().numpy()


class IQS(Metric):
    """
    Computes imputation quality score: https://doi.org/10.1371/journal.pone.0009697
    """
        
    # Set to True if the metric is differentiable else set to False
    is_differentiable: bool = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, Nclasses=2):
        super().__init__()
        self.add_state("cm", default=torch.zeros(Nclasses,Nclasses), dist_reduce_fx="sum")
        self.Nclasses = Nclasses

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        for class_idx in range(self.Nclasses):
            class_pos = target == class_idx
            self.cm[:,class_idx] += torch.FloatTensor([(preds[class_pos]==idx).sum() for idx in range(self.Nclasses)]).to(self.device)

    def compute(self):
        """
        Divide correct predictions by all predictions (ignoring the masked ones)
        """
        P0 = torch.diagonal(self.cm).sum()/self.cm.sum()
        Pc = (self.cm.sum(1)*self.cm.sum(0)).sum()/(self.cm.sum()**2)
        IQS = (P0-Pc)/(1-Pc)
        return IQS.item()
