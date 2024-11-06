import pandas as pd

dataset = pd.read_csv('dataset.csv')


attributes_info = {}

for column in dataset.columns:
    unique_values = dataset[column].nunique()  
    dtype = dataset[column].dtype              
    
    # Tentukan jenis atribut
    if unique_values <= 10 or dtype == 'object':
        attribute_type = 'Kategorik'
        unique_values_list = dataset[column].unique()  
    else:
        attribute_type = 'Numerik'
        unique_values_list = None 
    
    # Simpan informasi dalam dictionary
    attributes_info[column] = {
        'attribute_type': attribute_type,
        'unique_count': unique_values,
        'dtype': dtype,
        'unique_values_list': unique_values_list if attribute_type == 'Kategorik' else 'N/A'
    }


for attr, info in attributes_info.items():
    print(f"Atribut: {attr}")
    print(f"  - Tipe Atribut: {info['attribute_type']}")
    print(f"  - Jumlah Nilai Unik: {info['unique_count']}")
    print(f"  - Tipe Data: {info['dtype']}")
    if info['attribute_type'] == 'Kategorik':
        print(f"  - Nilai Unik: {info['unique_values_list']}")
    print()  

