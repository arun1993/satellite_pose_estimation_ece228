from utils import PyTorchSatellitePoseEstimationDataset
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from submission import SubmissionWriter

""" 
    Example script demonstrating training on the SPEED dataset using PyTorch.
    Usage example: python pytorch_example.py --dataset [path to speed] --epochs [num epochs] --batch [batch size]
"""


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

            # batch loop
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().cuda())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    return model


def evaluate_model(model, dataset, device, submission_writer, batch_size, real=False,):

    """ Function to evaluate model on \'test\' and \'real_test\' sets, and collect pose estimations to a submission."""

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model.eval()
    for inputs, filenames in dataloader:
        with torch.set_grad_enabled(False):
            inputs = inputs.to(device)
            outputs = model(inputs)

        q_batch = outputs[:, :4].cpu().numpy()
        r_batch = outputs[:, -3:].cpu().numpy()

        append = submission_writer.append_real_test if real else submission_writer.append_test
        for filename, q, r in zip(filenames, q_batch, r_batch):
            append(filename, q, r)
    return


def main(speed_root, epochs, batch_size):

    """ Preparing the dataset for training, setting up model, running training, and exporting submission."""

    # Processing to match pre-trained networks
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Loading training set, using 20% for validation
    full_dataset = PyTorchSatellitePoseEstimationDataset('train', speed_root, data_transforms)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * .8),
                                                                              int(len(full_dataset) * .2)])
    datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Getting pre-trained model and replacing the last fully connected layer
    initialized_model = models.resnet18(pretrained=True)
    
    for param in initialized_model.parameters():
        param.require_grad = False
   
    num_ftrs = initialized_model.fc.in_features
    initialized_model.fc = torch.nn.Linear(num_ftrs, 7)
    initialized_model = initialized_model.to(device)  # Note: we are finetuning the model (all params trainable)

    # Setting up the learning process
    criterion = torch.nn.MSELoss()
    sgd_optimizer = torch.optim.SGD(initialized_model.parameters(), lr=0.001, momentum=0.9)  # all params trained
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(sgd_optimizer, step_size=7, gamma=0.1)

    # Training
    trained_model = train_model(initialized_model, exp_lr_scheduler, sgd_optimizer, criterion,
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
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
    parser.add_argument('--epochs', help='Number of epochs for training.', default=20)
    parser.add_argument('--batch', help='number of samples in a batch.', default=32)
    args = parser.parse_args()

    main(args.dataset, int(args.epochs), int(args.batch))
