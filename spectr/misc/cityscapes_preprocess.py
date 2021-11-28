"""Preprocessing script to apply onto all (image, label) pairs in the Cityscapes Dataset."""

import glob
import pdb
from concurrent.futures import ProcessPoolExecutor

import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode, Resize

from spectr.src.utils import get_classId_from_rgb

def process_individual_image(img_path):
    print(f"Processing image: {img_path}")
    resize = Resize((256, 512), interpolation=InterpolationMode.NEAREST)
    # expectedLabelFilePath = img_path.replace("leftImg8bit", "gtFine", 1)
    # labels_path = expectedLabelFilePath.replace("leftImg8bit", "gtFine_color", 1)
    image = resize(read_image(img_path))
    img_name = img_path.split("/")[-1].replace(".png", "")
    directory = img_path.split("/")[-3]
    final_path = f"/coc/scratch/aahluwalia30/cityscapes_preprocessed/leftImg8bit/{directory}/{img_name}.pt"
    torch.save(image, final_path)
def process_individual_label(labels_path):
    #pdb.set_trace()
    print(f"Processing label: {labels_path}")
    resize = Resize((256, 512), interpolation=InterpolationMode.NEAREST) 
    label = resize(read_image(labels_path))
    label = label[:-1, :, :]
    label_classIds = torch.zeros(label.shape[1], label.shape[2])
    for row in range(label.shape[1]):
        for col in range(label.shape[2]):
            label_classIds[row, col] = get_classId_from_rgb(label[:, row, col])
    labels_name = labels_path.split("/")[-1].replace(".png", "")
    directory = labels_path.split("/")[-3]
    final_path = f"/coc/scratch/aahluwalia30/cityscapes_preprocessed/gtFine/{directory}/{labels_name}.pt"
    torch.save(label_classIds, final_path)
if __name__ == "__main__":
    pdb.set_trace()
    image_paths = glob.glob("/srv/datasets/cityscapes/leftImg8bit/*/*/*.png")
    labels_path = glob.glob("/srv/datasets/cityscapes/gtFine/*/*/*_color.png")
    #process_individual_label(labels_path[0])
    with ProcessPoolExecutor(max_workers = 64) as executor:
        print("Processing labels")
        #result_img = [executor.submit(process_individual_image, img) for img in image_paths]
        result_label = [executor.submit(process_individual_label, label) for label in labels_path]