import os


class DatasetCatalog:
    DATASETS = {
        'voc_mydata_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_mydata_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
    }

    @staticmethod
    def get(cfg, name):
        root = cfg["datasets"]["path"]
        if "voc" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root, attrs["data_dir"]),
                ann_file=os.path.join(root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
