# -*- coding: utf-8 -*-
import numpy as np
from util import load_model
import matplotlib.pyplot as plt
from torchvision import utils
import argparse


def loadModel(model_path):
    model = load_model(model_path)
    for params in model.parameters():
            params.requires_grad = False
    model.eval()
    return model

def getFeatures(model, layer):
    features = model.features[layer].weight.data.clone()
    return features

def plot(tensor, output, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig(output)
    
def visualize(model_path, layer, output):
    model = loadModel(model_path)
    features = getFeatures(model, layer)
    plot(features, output, ch=0, allkernels=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Saving model features")
    
    parser.add_argument('-m', '--model-path', default='checkpoint_alexnet.pth.tar',
                        help="Path of the saved model.")
    parser.add_argument('-l', '--layer', default=14,
                        help="Layer to be visualized.")
    parser.add_argument('-o', '--output', default='features.png',
                        help="Output path of the visualized features.")
    
    args = parser.parse_args()
    
    model_path = args.model_path
    layer = args.layer
    output = args.output
    visualize(model_path, layer, output)
