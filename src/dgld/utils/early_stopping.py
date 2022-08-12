import torch 
import numpy as np
import copy

class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.

    Parameters
    ----------
    early_stopping_rounds : int, optional
        Start early stopping after early_stopping_rounds, by default 0
    patience : int, optional
        How long to wait after last time loss improved, by default 7
    verbose : bool, optional
        If True, prints a message for each loss improvement, by default False
    delta : int, optional
        Minimum change in the monitored quantity to qualify as an improvement, by default 0
    check_finite : bool, optional
        When set ``True``, stops training when the monitor becomes NaN or infinite, by default True
    
    Examples
    -------
    >>> early_stop = EarlyStopping() 
    >>> for epoch in range(num_epoch):
    >>>     res = model(data) 
    >>>     loss = torch.mean(torch.pow(res - label,2))
    >>>     opt.zero_grad()
    >>>     loss.backward()
    >>>     opt.step()
    >>>     early_stop(loss,model)
    >>>     if early_stop.isEarlyStopping():
    >>>         print(f"Early stopping in round {epoch}")
    >>>         break
    """
    def __init__(self,early_stopping_rounds = 0, patience=7, verbose=False, delta=0,check_finite=True):        

        self.early_stop_rounds = early_stopping_rounds
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.__early_stop = False
        self.loss_min = torch.tensor(torch.inf)
        self.delta = delta
        self.rounds = 0
        self.check_finite = check_finite
        self.__best_parameters = None
    @property
    def early_stop(self):
        """
        Returns
        -------
        bool
            Return whether early stopping.
        """
        return self.__early_stop

    @property
    def best_paramenters(self):
        """

        Returns
        -------
        OrderedDict
            The model.state_dict() of minimal loss
        """
        return self.__best_parameters

    def isEarlyStopping(self):
        return self.early_stop

    def __call__(self, loss, model=None):
        """
        The function to judge early stopping

        Parameters
        ----------
        loss : float
            The loss of a epoch.
        model : torch.nn.modules
            The model
        """
        if isinstance(loss,torch.Tensor):
            loss = loss.cpu().item()
        if self.check_finite and not np.isfinite(loss):
            self.__early_stop = True
            if self.verbose:
                print(f"Loss = {loss} is not finite.")
        else:
            self.rounds += 1  
            if loss > self.loss_min - self.delta:
                if self.rounds > self.early_stop_rounds:
                    self.counter += 1
                    if self.verbose:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.__early_stop = True
            else:
                if self.verbose:
                    print(f'Rounds : {self.rounds} Validation loss decreased ({self.loss_min:.6f} --> {loss:.6f}).')
                self.loss_min =loss
                self.counter = 0
                if model is not None:
                    self.save_best_parameters(model)

        if self.__early_stop and self.verbose:
            print(f"Previous best loss was {self.loss_min:.6f}. Signaling Trainer to stop")

    def save_best_parameters(self, model):
        """
        Saves model.state_dict() of the minimal loss

        Parameters
        ----------
        model : torch.nn.modules
            The model
        """
        self.__best_parameters = copy.deepcopy(model.state_dict())
        