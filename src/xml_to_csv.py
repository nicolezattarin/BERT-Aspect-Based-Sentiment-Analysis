import xml.etree.ElementTree as Xet
import pandas as pd
import shutil

def aspectTerms(file_name, output_file):
    """
    convert xml file to csv file
    """
    rows = []
    # print("processing: ", file_name)
    # Parsing the XML file
    xmlparse = Xet.parse(file_name)
    root = xmlparse.getroot()
    for child in root: # childs is a space
        for elem in child:   
            if elem.tag == 'text':
                text = elem.text
            elif elem.tag == 'aspectTerms':
                for aspect in elem:
                    space = []
                    space.append(child.attrib['id'])
                    space.append(text)
                    space.append(aspect.attrib['term'])
                    space.append(aspect.attrib['polarity'])
                    space.append(aspect.attrib['from'])
                    space.append(aspect.attrib['to'])
                    rows.append(space)
    cols = ['id', 'text', 'term', 'polarity', 'from', 'to']

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_file, index=False)

def aspectCategories(file_name, output_file):
    rows = []
    # print("processing: ", file_name)
    # Parsing the XML file
    xmlparse = Xet.parse(file_name)
    root = xmlparse.getroot()
    for child in root: # childs is a space
        for elem in child:   
            if elem.tag == 'text':
                text = elem.text
            elif elem.tag == 'aspectCategories':
                for aspect in elem:
                    space = []
                    space.append(child.attrib['id'])
                    space.append(text)
                    space.append(aspect.attrib['category'])
                    space.append(aspect.attrib['polarity'])
                    rows.append(space)
    cols = ['id', 'text', 'category', 'polarity']

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_file, index=False)


def main():
    path = '../data/Restaurants_Train.xml'
    aspectTerms(path, '../data/Restaurants_Train_apectTerms.csv')
    aspectCategories(path, '../data/Restaurants_Train_apectCategories.csv')

    test = '../data/ABSA_Gold_TestData/Restaurants_Test_Gold.xml'
    aspectTerms(test, '../data/ABSA_Gold_TestData/Restaurants_Test_Gold_apectTerms.csv')
    aspectCategories(test, '../data/ABSA_Gold_TestData/Restaurants_Test_Gold_apectCategories.csv')

if __name__ == '__main__':
    main()