import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = image_size / feature_size

    # grid centers
    grid = np.arange(stride / 2, image_size, stride)
    
    anchors = []

    for y in grid:
        for x in grid:
            for s in scales:
                for ar in aspect_ratios:
                    w = s * np.sqrt(ar)
                    h = s / np.sqrt(ar)
                    anchors.append([x-w/2, y-h/2, x+w/2, y+h/2])

    return anchors