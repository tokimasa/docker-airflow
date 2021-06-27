import mysql.connector
from mysql.connector import Error
import pandas as pd

def db_connect():
	try:
		# Connect MySQL/MariaDB database
		connection = mysql.connector.connect(
			# host='localhost',
			host='172.31.224.1',
			database='demo',
			user='root',
			password='0000')

		# Query DB
		cursor = connection.cursor()
		cursor.execute("SELECT * FROM wine_quality;")

		# Fetch all data
		table_rows = cursor.fetchall()
		df = pd.DataFrame(table_rows, columns=cursor.column_names)
		print('DB Access Success!')
		return df
	except Error as e:
		print("connection failed：", e)

	finally:
		if (connection.is_connected()):
			cursor.close()
			connection.close()
			print("connection closed")
			
def data_dumping(df, saving_type='csv'):
	if saving_type == 'csv':
		df.to_csv(r'..\data\download_data.csv', index=False)
		print('Data saved!')
	elif saving_type == 'mysql':
		## To Do ...
		print('Need to load data to mysql~')

def ingest_data():
	df = db_connect()
	data_dumping(df)

if __name__=="__main__":
	ingest_data()