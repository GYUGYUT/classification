
from torchvision import models
from report_save import *
from paser_args import *
from confusion_matrix import *
from torchvision import models
import torchsummary

from torchvision import models
import torch.nn as nn
import torch
from getmodel import *

model = getModel("resnext101",5)
device = torch.device("cpu")
model.to(device)
model.eval()

torchsummary.summary(model, (3, 512,512),device=device)
