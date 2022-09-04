class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


config = dict2(**{
        "mode":         "CNN",
        "dataset":      "BDD",
        "update_rule":  "adam",
        "CNNmodel":     "NVIDIA",
        "h5path":       "./data/processed_SAX_cropped/", #TODO Change Whether SAX or BDD, also change dataloader_CNN_course2 and the 10 and 1 iterators
        "imgRow":       90,
        "imgCol":       160,
        "imgCh":        3,
        "resizeFactor": 1,
        "batch_size":   400,
        "lr":           3e-5,
        "timelen":      4,
        "use_smoothing": None,
        "alpha":        1.0,    # coefficient for exp smoothing
        "UseFeat":      False,
        "maxiter":      60100,
        "save_steps":   250, #50 od. 20
        "val_steps":    50, #10
        "model_path":   "./model/CNN/",
        "pretrained_model_path": None,
        "gpu_fraction": 0.8 }) 
