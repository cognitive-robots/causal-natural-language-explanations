class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

config = dict2(**{
        "mode":         "VA",
        "dataset":      "BDD",
        "update_rule":  "adam",
        "CNNmodel":     "NVIDIA",
        "h5path":       "./data/processed_full/", #TODO Change Whether SAX or BDDX, also change dataloader_VA_course2 and the 10 and 1 iterator
        "imgRow":       90,
        "imgCol":       160,
        "imgCh":        3,
        "resizeFactor": 1,
        "n_epoch":      100000,
        "epoch":        5, #20,
        "maxiter":      180100, #140100
        "lr":           1e-5, #3e-4 (run 1), 1e-4
        "save_steps":   50, #1000, BDDx 50
        "val_steps":    50, #100, BDDX 50
        "model_path":   "./model/VA/",
        "model_path_Gen":   "./model/LSTM_Gen/",
        "pretrained_model_path": None,
        "pretrained_model_path_Gen": None,
        "test_replay":  None,
        "use_smoothing": None, #"Exp",
        "UseFeat":      True,
        "alpha":        1.0,
        "dim_hidden":   1024,
        "dim_hidden_Gen": 512, #1024
        "batch_size":   32, #16, 4, #20    #2,
        "batch_size_gen": 32,
        "timelen":      20+3,  #20+3, 
        "ctx_shape":    [240,64],
        "alpha_c":      100.0,
        "dict_size":    1300,
        "subsample":    1,
        "batches_per_dataset": 464,
        "batches_per_dataset_v": 58,
        "batches_per_dataset_t": 1818})

