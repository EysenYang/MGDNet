from __future__ import print_function
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.utils.data.dataset
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import sys
import cv2

torch.set_num_threads(8)

class TestDataset(torch.utils.data.dataset.Dataset):
    """Loading data from input file formatted as csv.
    """
    def __init__(self, images, transform=None):
        self.transform = transform
        self.images = images

    def __getitem__(self, i):
        path = self.images[i]
        
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)

        return img, 0

    def __len__(self):
        return len(self.images)


def predict(model, images, mean, std):
    model.eval() 

    normalize = transforms.Normalize(mean=mean, std=std)
    dataset_test = TestDataset(
            images,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=4)
    
    probs = []
    
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            data = data.cuda()
            with torch.no_grad():
                prob  = F.softmax(model(data), dim=1).tolist()
                probs.extend(prob)

    return np.asarray(probs)

def predict_video(model, video_paths, mean, std):

    model.eval()
    
    tfs = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])

    
    scores = []
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        probs = []
        frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                break
            frame = Image.fromarray(frame)
            frames.append(tfs(frame))
            if len(frames) == 16:
                data = torch.cat(frames).cuda()
                with torch.no_grad():
                    output = F.softmax(model(data), dim=1).data.cpu().numpy()[:,1].tolist()
                    probs.extend(output)
                frames = []
        cap.release()
    
        if len(frames) >= 2:
            data = torch.cat(frames).cuda()
            with torch.no_grad():
                output = F.softmax(model(data), dim=1).data.cpu().numpy()[:,1].tolist()
                probs.extend(output)
            frames = []

        probs = np.asarray(probs)
        
        weighted_mean = np.average(probs, weights=probs)
        mean = np.mean(probs)
        #top3_prob = np.sort(probs)[-3]
        
        a = [video_path, weighted_mean, mean]
        scores.append(a)
        #print(a)
        
    df = pd.DataFrame(scores, columns=['avi', 'image_weighted_mean_prob', 'image_mean_prob'])

    return df

def resnet34_image_model(num_classes, checkpoint):
    state_dict = torch.load(checkpoint, map_location='cpu')
    model = models.resnet34(num_classes=num_classes).cuda()
    model.load_state_dict(state_dict['model'])
    model.eval()
    return model


if __name__ == '__main__':
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    model = resnet34_image_model(6, "./resnet34_Ep120_wd1e4/model_119.pth")
    model.eval()


    tjmuch_mean=[0.485, 0.456, 0.406]
    tjmuch_std =[0.229, 0.224, 0.225]

    data_file = "./val.csv"
    result_data = pd.read_csv(data_file)
    internal_df = predict(model, result_data.image_name.to_list(), tjmuch_mean, tjmuch_std)

    result_data = result_data.assign(image_p0=internal_df[:,0])
    result_data = result_data.assign(image_p1=internal_df[:,1])
    result_data = result_data.assign(image_p2=internal_df[:,2])
    result_data = result_data.assign(image_p3=internal_df[:,3])
    result_data = result_data.assign(image_p4=internal_df[:,4])
    result_data = result_data.assign(image_p5=internal_df[:,5])
    result_data.to_csv("./predict_result.csv")


