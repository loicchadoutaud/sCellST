RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]
MAPPING_DICT = {
    "Background": 0,
    "Neoplastic": 1,
    "Inflammatory": 2,
    "Connective": 3,
    "Dead": 4,
    "Non - Neoplastic": 5,
}
COLOR_DICT = {
    0: (255, 255, 255),  # blanc
    1: (255, 0, 0),  # rouge
    2: (0, 255, 0),  # vert
    3: (0, 0, 255),  # bleu
    4: (127, 0, 255),  # violet
    5: (255, 128, 0),  # orange
}

