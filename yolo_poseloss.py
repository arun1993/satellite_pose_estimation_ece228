from utils import PyTorchSatellitePoseEstimationDataset

import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from submission import SubmissionWriter
import torch.nn.functional as F
import argparse
import torch.nn as nn

from yolo_models import *

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
#mp = mp.get_context('spawn')
#torch.multiprocessing.set_start_method('spawn')
""" 
    Example script demonstrating training on the SPEED dataset using PyTorch.
    Usage example: python pytorch_example.py --dataset [path to speed] --epochs [num epochs] --batch [batch size]
"""
from torchvision.models.resnet import BasicBlock , Bottleneck

print(Bottleneck)
#model = MyResnet2(BasicBlock, [3, 4, 6, 3], 1000)
#x = Variable(torch.randn(1, 3, 224, 224))
#_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,**kwargs)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}
  
class myyoloModel(nn.Module):
    def __init__(self,model):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(myyoloModel, self).__init__()
       
        #resnet = models.resnet50(pretrained=True)
        self.model = model
        
        print("YOLO",len(list(list(self.model.children())[0].children())))
        
        modules = list(list(self.model.children())[0].children())[0:-1]  # delete the last YOLO layer.
        self.model = nn.Sequential(*modules)
        
        
        self.fc_orie = nn.Linear(2048, 4)
        self.fc_pos = nn.Linear(2048, 3)
        
        # self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, x):
        
        # with torch.no_grad():
        #     features = self.resnet(images)
        # features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))

        x = self.model(x)  # need gradient
        output = x.view(x.size(0), -1)
        
        pos_output = self.fc_pos(output)
        orie_output = self.fc_orie(output)
        
        return orie_output,pos_output
    
        
#self.sx = 0
#self.sq = -6.25 # or -3.0 if learn_beta = True
    
def PoseNetLoss(pred_x,pred_q,target_x,target_q, sx = 0.0, sq = -6.25):

    #crit1 = torch.nn.L1loss()
    #crit2 = torch.nn.L1loss()

    #print(target_x.size())
    pred_q = F.normalize(pred_q, p=2, dim=1)
    
    sx = nn.Parameter(torch.Tensor([sx]).cuda(), requires_grad=False)
    sq = nn.Parameter(torch.Tensor([sq]).cuda(), requires_grad=False)
    
    loss_x = F.l1_loss(pred_x, target_x)
    loss_q = F.l1_loss(pred_q, target_q)

    #loss_x = F.mse_loss(pred_x, target_x)
    #loss_q = F.mse_loss(pred_q, target_q)

    #print(type(loss_x),loss_x.device)

    loss = torch.exp(-sx)*loss_x + sx \
         + torch.exp(-sq)*loss_q + sq

    
    return loss.cuda().float(), loss_x.cuda().float(), loss_q.cuda().float()

    
    
def train_model(model, scheduler, optimizer, criterion, dataloaders, device, dataset_sizes, num_epochs):

    """ Training function, looping over epochs and batches. Return the trained model. """

    # epoch loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0
            running_pose_loss = 0.0
            running_joint_loss = 0.0

            i = 0
            # batch loop
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'): # gradient enabled only for train mode
                    
                    #out_orie,out_pos = model(inputs)
                    outputs = model(inputs)
                    #print(outputs[0].size(),outputs[1].size())
                    #print(labels[0])
                    #print(outputs[0].size())
                    tog_outputs = torch.cat((outputs[0],outputs[1]),-1)
                    #print(tog_outputs.size(),labels.size())
                    loss = criterion(tog_outputs, labels.float().cuda())
                    
                    #print("labels",labels.size())
                    poseLoss,posLoss,orieLoss = PoseNetLoss(outputs[1],outputs[0],labels.float()[:,4:],labels.float()[:,:4],\
                                               sx = 0.0, sq = -1.5)
                    
                    #print(type(loss),type(poseLoss))
                    #print(loss.device,poseLoss.device)
                    #print(loss+poseLoss)
                    
                    #final_loss = torch.Tensor([(poseLoss+loss).item()]).cuda()
                    #print("fi",final_loss)
                    
                    if phase == 'train':
                        (poseLoss+loss).backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_pose_loss += poseLoss.item() * inputs.size(0)
                running_joint_loss += (posLoss.item() + orieLoss.item()) * inputs.size(0)
                
                i += 1
                
                if i%100==0:
                    print(" Active!")
                
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            overall_epoch_loss = running_pose_loss / dataset_sizes[phase]
            print('{} Overall Loss: {:.4f}'.format(phase, overall_epoch_loss))
            
            overall_joint_loss = running_joint_loss / dataset_sizes[phase]
            print('{} Overall Joint Loss: {:.4f}'.format(phase, overall_joint_loss))

    return model


def evaluate_model(model, dataset, device, submission_writer, batch_size, real=False,):

    """ Function to evaluate model on \'test\' and \'real_test\' sets, and collect pose estimations to a submission."""

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model.eval()
    for inputs, filenames in dataloader:
        with torch.set_grad_enabled(False):
            inputs = inputs.to(device)
            outputs = model(inputs)

        #q_batch = outputs[:, :4].cpu().numpy()
        #r_batch = outputs[:, -3:].cpu().numpy()
        q_batch = outputs[0].cpu().numpy()
        r_batch = outputs[1].cpu().numpy()

        append = submission_writer.append_real_test if real else submission_writer.append_test
        for filename, q, r in zip(filenames, q_batch, r_batch):
            append(filename, q, r)
    return


def main(speed_root, epochs, batch_size):

    """ Preparing the dataset for training, setting up model, running training, and exporting submission."""

    # Processing to match pre-trained networks
    
    # orig size = 1920×1200  pixels
    data_transforms = transforms.Compose([
        transforms.Resize((416, 416)), ## YOLO input size !!
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Loading training set, using 20% for validation
    
    full_dataset = PyTorchSatellitePoseEstimationDataset('train', speed_root, data_transforms)
    
    # pytorch utility to create random splits
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * .8),int(len(full_dataset) * .2)])
    
    print("train object",train_dataset)
    
    datasets = {'train': train_dataset, 'val': val_dataset}
    
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=8)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''
    MODIFY ARCHITECTURE HERE
    '''
    
    # Getting pre-trained model and replacing the last fully connected layer
    
    #initialized_model = models.resnet50(pretrained=True)
    #num_ftrs = initialized_model.fc.in_features
        
   ## Build 2 fc layers for position and orientation seperately, calculate different loss and backprop together 

    #initialized_model = myorigModel(Bottleneck, [3, 4, 6, 3],1000)
     # Initiate model
        
    model = Darknet("yolov3.cfg")
    #model.apply(weights_init_normal)

    model.load_darknet_weights("./darknet53.conv.74")
            
    new_model = myyoloModel(model)
    
    #print(initialized_model)
    
    ## Now try loading the weights thing:
    #state_dict = load_state_dict_from_url(model_urls['resnet50'])
    #initialized_model.load_state_dict(state_dict)
    
    #new_model = myModel(initialized_model).model
    
    print("New",new_model)
    new_model = new_model.to(device)  # Note: we are finetuning the model (all params trainable)
    
    
    # Setting up the learning process
    #criterion = torch.nn.MSELoss() # real output values
    criterion = torch.nn.L1Loss() 
    
    ### USE EVERYTHING POSENET INSPIRED !!!!!!!!!!!!!!!!!
    # crit1, crit2 should be L1 loss
    
    sgd_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.001, momentum=0.9)  # all params trained
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(sgd_optimizer, step_size=7, gamma=0.1)

    # Training
    trained_model = train_model(new_model, exp_lr_scheduler, sgd_optimizer, criterion,
                                dataloaders, device, dataset_sizes, epochs)

    # Generating submission
    submission = SubmissionWriter()
    test_set = PyTorchSatellitePoseEstimationDataset('test',  speed_root, data_transforms)
    real_test_set = PyTorchSatellitePoseEstimationDataset('real_test',  speed_root, data_transforms)

    evaluate_model(trained_model, test_set, device, submission, batch_size, real=False)
    evaluate_model(trained_model, real_test_set, device, submission, batch_size, real=True)

    submission.export(suffix='pytorch_example')
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='./speed')
    parser.add_argument('--epochs', help='Number of epochs for training.', default=5)
    parser.add_argument('--batch', help='number of samples in a batch.', default=32)
    args = parser.parse_args()

    main(args.dataset, int(args.epochs), int(args.batch))
