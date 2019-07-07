import pdb
from random import shuffle
import shutil
import xml.etree.cElementTree as ET
from collections import defaultdict


def create_xml_file(dataroot, filename, obj_name, folder, ball_details):
    if len(ball_details)==1:
        width, height = ball_details[0][0], ball_details[0][1]
        xmin, ymin =  ball_details[0][2], ball_details[0][3]
        xmax, ymax = ball_details[0][4], ball_details[0][5]

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
        bndbox = ET.SubElement(obj, 'bndbox1')
        ET.SubElement(bndbox, 'xmin').text = xmin
        ET.SubElement(bndbox, 'ymin').text = ymin
        ET.SubElement(bndbox, 'xmax').text = xmax
        ET.SubElement(bndbox, 'ymax').text = ymax
        tree = ET.ElementTree(root)
        tree.write('{}{}.xml'.format(dataroot, filename))

    else:
        width, height = ball_details[0][0], ball_details[0][1]
        xmin1, ymin1 =  ball_details[0][2], ball_details[0][3]
        xmax1, ymax1 = ball_details[0][4], ball_details[0][5]

        xmin2, ymin2 =  ball_details[1][2], ball_details[1][3]
        xmax2, ymax2 = ball_details[1][4], ball_details[1][5]

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
        bndbox = ET.SubElement(obj, 'bndbox1')
        ET.SubElement(bndbox, 'xmin').text = xmin1
        ET.SubElement(bndbox, 'ymin').text = ymin1
        ET.SubElement(bndbox, 'xmax').text = xmax1
        ET.SubElement(bndbox, 'ymax').text = ymax1

        bndbox1 = ET.SubElement(obj, 'bndbox2')
        ET.SubElement(bndbox1, 'xmin').text = xmin2
        ET.SubElement(bndbox1, 'ymin').text = ymin2
        ET.SubElement(bndbox1, 'xmax').text = xmax2
        ET.SubElement(bndbox1, 'ymax').text = ymax2

    tree = ET.ElementTree(root)
    tree.write('{}{}.xml'.format(dataroot, filename))



def prepare_dataset(dataroot, annotation_file):
    with open(dataroot + annotation_file, 'r') as f:
        lines = f.readlines()[6:]
        lines = [line.strip().split('|') for line in lines]
        filenames_d = defaultdict(list)
        for line in lines:
            filenames_d[line[1]].append(line)
        filenames = list(filenames_d.keys())
        shuffle(filenames)
        split = int(0.6*len(filenames))
        training = filenames[:split]
        test = filenames[split:]
        filename_balls = {}
        # for line in training:
        for fn in training:
            for line in filenames_d[fn]:
                label, filename = line[0], line[1]
                if label=='label::ball':
                    ball_details = [line[2], line[3], line[4], line[5], line[6], line[7]]
                    if filename in filename_balls:
                        filename_balls[filename].append(ball_details)
                    else:
                        filename_balls[filename] = [ball_details]

        for filename, ball_details in filename_balls.items():
            source_filename = '{}multiballsamples/{}'.format(dataroot, filename)
            shutil.copy(source_filename, dataroot + '/train_cnn/' + filename)
            folder = 'train_cnn'
            obj_name = 'ball'
            dataroot_train = dataroot + 'train_cnn/'
            create_xml_file(dataroot_train, filename[:-4], obj_name, folder, ball_details)

        filename_balls_test = {}
        for fn in test:
            for line in filenames_d[fn]:
                label, filename = line[0], line[1]
                if label=='label::ball':
                    ball_details = [line[2], line[3], line[4], line[5], line[6], line[7]]
                    if filename in filename_balls_test:
                        filename_balls_test[filename].append(ball_details)
                    else:
                        filename_balls_test[filename] = [ball_details]

        for filename, ball_details in filename_balls_test.items():
            source_filename = '{}multiballsamples/{}'.format(dataroot, filename)
            shutil.copy(source_filename, dataroot + '/test_cnn/' + filename)
            folder = 'test_cnn'
            obj_name = 'ball'
            dataroot_test = dataroot + 'test_cnn/'
            create_xml_file(dataroot_test, filename[:-4], obj_name, folder, ball_details)


if __name__=='__main__':

    dataroot = '../SoccerDataMulti/'
    annotation_file1 = 'export_ball_1508.txt'
    prepare_dataset(dataroot, annotation_file1)