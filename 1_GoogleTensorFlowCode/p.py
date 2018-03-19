'''
months = [
	'january',
	'february',
	'march',
	'april',
	'may',
	'june',
	'july',
	'august',
	'sepyember',
	'october',
	'novenber',
	'decenber',
]
endings = ['st','nd','rd'] + 17*['th']\
          + ['st','nd','rd'] + 7*['th']\
		  +['st']
year = input ('Year: ')
month = input ('month(1-12): ')
day = input ('Day(1-31): ')

month_munber = int (month)
day_munber = int (day)

month_mane = months[month_munber-1]
ordinal = day + endings[day_munber-1]

print (month_mane + ' ' + ordinal + ' ' +year)
'''

'''
numbers = [1,2,3,4,5,6,7,8,9,10]
print(numbers[:])


url = input ('enter the url: ')
domain = url [12:-4]

print ('domain: '+ domain)




print (numbers[6::-3])
'''
'''
sequence = [None]*20
print (sequence)
'''
'''

users = ['min' , 'foo' , 'bar']
print (input('输入名字：') in users)

'''

'''
format="hello,%s,%s,enouph for ya"
values=('world','hot')
print(format%values)

import string 

table = str.maketrans('cs','kz')

print(len(table))
print(table[97:123])
print(str.maketrans('','')[91:123])
'''

#phonebook = {'Alice':'121588','Beth':'184878','Cecil':'588',}
'''
items=[(name,Alice),(phone,548188)]
d=dict(items)
print(d)
print(len(d))
print(d[name])
d[name]=Mike
print(d[name])
'''

'''
pepole= {
	'Alice' :{
		'phone' : '12188',
		'addr' : 'Foo drive 23'
	},
	'Beth' :{
		'phone' : '7881818',
		'addr' : 'FBAR river 73'
	},
	'Cecil' :{
		'phone' : '88947',
		'addr' : 'Baz kjin 75'
	}
}

labels = {
	'Phone' : 'Phone Munber',
	'addr' : 'address'
}
name = input ('name: ')
request = input ('PhoneNumber(p) or adress(a)? ')

if request == 'p' : key = 'phone'
if request == 'a' : key = 'addr'

if name in pepole : \
print("%s's , %s is %s" %(name, labels[key],pepole[name][key]))
'''
'''
phonebook = {'Alice':'121588','Beth':'184878','Cecil':'588'}

print("Cecil 's phone number is %(Cecil)s." %phonebook)

import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier()

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)


import pandas as pd
print(pd.__version__)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities=pd.DataFrame({ 'City name': city_names, 'Population': population })

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())

print(california_housing_dataframe.head())

print(california_housing_dataframe.hist('housing_median_age'))


import numpy as np

np.log(population)

population.apply(lambda val: val > 1000000)

cities['Area square miles'] = pd.Series([46.87,46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & \
										cities['City name'].apply(lambda name: name.startswith('San'))

print (cities.index)

cities = cities.reindex([2,5,1,4,9])
print ( cities )

cities=cities.reindex(np.random.permutation(cities.index))

print(cities)
cities=cities.reindex(np.random.permutation(cities.index))

print(cities)
cities=cities.reindex(np.random.permutation(cities.index))

print(cities)
cities=cities.reindex(np.random.permutation(cities.index))

print(cities)
'''
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")


california_housing_dataframe = california_housing_dataframe.reindex(\
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
print(california_housing_dataframe)

# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Define the label.
targets = california_housing_dataframe["median_house_value"]


# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
         ds = ds.shuffle(buffer_size=10000)
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

_ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature, targets),steps=100)
# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print ("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print ("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print ("Min. Median House Value: %0.3f" % min_house_value)
print ("Max. Median House Value: %0.3f" % max_house_value)
print ("Difference between Min. and Max.: %0.3f" % min_max_difference)
print ("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print (calibration_data.describe())

# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()






'''
format="Pi with three decimals: %.3f"
from math import pi
print ( format % pi)
'''