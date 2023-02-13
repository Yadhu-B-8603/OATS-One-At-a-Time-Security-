import mysql.connector
import serial
import time

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="DC",
  database="rfid"
)
A="13 F1 CF 92"
a=0


mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM students")

myresult = mycursor.fetchall()



list_string = []
ser = serial.Serial()
ser.baudrate = 9600
ser.port = 'COM10'
data = ""
ser.open()
while True:
    # state=(input("Enter the state of LED")).encode()
    # ser.write(state)
    while (ser.inWaiting() == 0):
        # print("1")
        pass
    try:
        data += ser.read().decode()
    except:
        pass
    if (len(data) == 12):
        #print(data)
        for x in myresult:
          V=list(x);
          #print(V)
          Y=str(V[0])
          #print(Y)
          if(Y==data):
            a=1
            print(V[1])
            print(Y)
            
            break;
    
          else:
          
            a=2
            
    
        if a==1:
            ser.write(b'1');
            print("Verified");

  
        if a==2:
            ser.write(b'0');
            print("Not Verified");
        
        
        data = ""



ser.close()
