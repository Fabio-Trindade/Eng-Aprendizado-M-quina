import mysql.connector
class MySQL:
    def __init__(self,host,user,password,database):
        self.password = password
        self.host = host
        self.user = user
        self.database =  database
        
    def get_connection(self):
        try:
            connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database )
            if connection.is_connected():
                return connection
        except mysql.connector.Error as err:
            print("Erro ao conectar ao banco de dados:", err)
        

    def execute_query(self, query:str):
        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        connection.close()
        return data
    
