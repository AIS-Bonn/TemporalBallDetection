import pdb
from arguments import opt
from random import shuffle
import shutil
import xml.etree.cElementTree as ET


def create_xml_file(dataroot, filename, obj_name, folder, width, height, xmin, ymin, xmax, ymax):
    root = ET.Element('annotation')
    folder = ET.SubElement(root, 'folder').text = folder
    doc = ET.SubElement(root, 'filename').text = filename

    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = 'visionlab'
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = width
    ET.SubElement(size, 'height').text = height
    ET.SubElement(size, 'depth').text = '3'
    obj = ET.SubElement(root, 'object')
    ET.SubElement(obj, 'name').text = obj_name
    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = xmin
    ET.SubElement(bndbox, 'ymin').text = ymin
    ET.SubElement(bndbox, 'xmax').text = xmax
    ET.SubElement(bndbox, 'ymax').text = ymax

    tree = ET.ElementTree(root)
    tree.write('{}{}.xml'.format(dataroot, filename))


def prepare_dataset(dataroot, annotation_file):
    with open(dataroot + annotation_file, 'r') as f:
        lines = f.readlines()[6:]
        lines = [line.strip().split('|') for line in lines]
        shuffle(lines)
        split = int(0.7*len(lines))
        training = lines[:split]
        test = lines[split:]
        for line in training:
            label, filename = line[0], line[1]
            target_filename = '{}_{}'.format(filename[:-4], annotation_file[:-4])
            source_filename = '{}{}/{}'.format(dataroot, annotation_file[:-4], filename)
            shutil.copy(source_filename, dataroot+'SoccerData1/train_cnn/'+target_filename+ filename[-4:])
            if label=='label::ball':
                width, height = line[2], line[3]
                xmin, ymin, xmax, ymax = line[4], line[5], line[6], line[7]
                folder = 'train_cnn'
                obj_name = 'ball'
                dataroot_train = dataroot + 'SoccerData1/train_cnn/'
                create_xml_file(dataroot_train, target_filename, obj_name, folder, width, height, xmin, ymin, xmax, ymax)

        for line in test:
            label, filename = line[0], line[1]
            target_filename = '{}_{}'.format(filename[:-4], annotation_file[:-4])
            source_filename = '{}{}/{}'.format(dataroot, annotation_file[:-4], filename)
            shutil.copy(source_filename, dataroot+'SoccerData1/test_cnn/'+target_filename+filename[-4:])
            if label=='label::ball':
                width, height = line[2], line[3]
                xmin, ymin, xmax, ymax = line[4], line[5], line[6], line[7]
                folder = 'test_cnn'
                obj_name = 'ball'
                dataroot_test = dataroot + 'SoccerData1/test_cnn/'
                create_xml_file(dataroot_test, target_filename, obj_name, folder, width, height, xmin, ymin, xmax, ymax)

if __name__=='__main__':

    dataroot = '../SoccerDataMulti/'
    annotation_file1 = 'export_ball_1508.txt'
    prepare_dataset(dataroot, annotation_file1)
    # annotation_file2 = 'imageset_431.txt'
    # prepare_dataset(dataroot, annotation_file2)
    # annotation_file3 = 'imageset_432.txt'
    # prepare_dataset(dataroot, annotation_file3)