import pickle
folder = 'data/five_years/five_years_by_type/'
for i in range(1, 10):
    filename = folder + 'type_' + str(i) + '.pkl'
    print('filename:', filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print(data)
# for index, row in data.iterrows():
#    print(index, row['YEAR'], row['QUESTION'], row['cluster'], sep='\t')
