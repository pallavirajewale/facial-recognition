import pickle

data = {"name": "John", "age": 30}

# Saving the data to a pickle file
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
