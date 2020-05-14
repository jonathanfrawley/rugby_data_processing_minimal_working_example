
# Requirements
Requires:

 - python 3
 - matplotlib
 - numpy
 - pandas
 - scipy

# Running


    python3 process.py <filename.csv>

This should output linear and angular acceleration graphs for each impact in a directory called "out".

# CSV Format


    t,Date,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z

where:
 - t is a timestamp to two decimal places
 - acc_{x,y,z} are accelerometer values in units of G 
 - gyr_{x,y,z} are gyroscope values in units of rad/s
