#!/usr/bin/python

import pandas as pd
import numpy as np


def prepareData():
    data = pd.read_csv("Customers.csv")
    #print(data[:10])

    dataF = data

    mapping = {'NaN' : 0, 'Healthcare' : 1, 'Engineer' : 2, 'Lawyer' : 3, 'Entertainment' : 4, 'Artist' : 5, 'Executive' : 6,
    'Doctor' : 7, 'Homemaker' : 8, 'Marketing' : 9}

    mapping2 = {'Male' : 0, 'Female' : 1}

    dataF = dataF.replace({'Profession': mapping})
    dataF = dataF.replace({'Gender': mapping2})

    dataF = dataF.drop(columns=['CustomerID'])

    dataF['Profession'] = dataF['Profession'].fillna(0)

    normalized_dataF = (dataF - dataF.min())/(dataF.max() - dataF.min())

    #print(normalized_dataF[:10])

    train_data = normalized_dataF[0:1600]
    dev_data = normalized_dataF[1600:1800]
    test_data = normalized_dataF[1800:]

    #print(f"Wielkość zbioru Customers: {len(data)} elementów")
    #print(f"Wielkość zbioru trenującego: {len(train_data)} elementów")
    #print(f"Wielkość zbioru walidującego: {len(dev_data)} elementów")
    #print(f"Wielkość zbioru testującego: {len(test_data)} elementów")

    #print(f" \nDane i wartości na temat zbioru: \n \n {normalized_dataF.describe()}")

    return train_data, dev_data, test_data
