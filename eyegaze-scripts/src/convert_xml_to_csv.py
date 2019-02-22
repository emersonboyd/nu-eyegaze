import os
import util
import glob
import pandas as pd

import xml.etree.ElementTree as ET


# Modified From:
# https://github.comr/datitran/raccoon_dataset/blob/master/xml_to_csv.py

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob('{}/*.xml'.format(path)):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    label_path = '{}/sign_labels'.format(util.get_resources_directory())
    for dataset_type in ['train', 'test']:
        label_path_current = '{}/{}'.format(label_path, dataset_type)
        xml_df = xml_to_csv(label_path_current)
        xml_df.to_csv('{}/{}.csv'.format(label_path, dataset_type), index=None)
        print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()


# main():
#     for i in [trainPath, testPath]:
#         image_path = i
#         folder = os.path.basename(os.path.normpath(i))
#         xml_df = xml_to_csv(image_path)
#         xml_df.to_csv('{}/' + folder + '.csv', index=None)
#         print('Successfully converted xml to csv.')
