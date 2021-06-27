import mysql.connector
from mysql.connector import Error
import pandas as pd
import os

def db_connect():
	try:
		# Connect MySQL/MariaDB database
		connection = mysql.connector.connect(
			# host='localhost',
			host='192.168.50.158',
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
		## for the local PC working dir, "docker-airflow\dags\scripts\data_ingestion.py" => "docker-airflow\dags\data\download_data.csv"
		# df.to_csv(r'..\data\download_data.csv', index=False)
		## for the airflow working dir, /usr/local/airflow/ , it needs to save in /usr/local/airflow/dags/ for the connection volume directory
		df.to_csv(r'./dags/data/download_data.csv', index=False)
		print('Data saved!')
		print(os.path.abspath(os.getcwd()))
		print(os.listdir(os.curdir))
		
	elif saving_type == 'mysql':
		## To Do ...
		print('Need to load data to mysql~')

def ingest_data():
	df = db_connect()
	data_dumping(df)

if __name__=="__main__":
	ingest_data()