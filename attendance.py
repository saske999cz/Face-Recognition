import csv
import datetime
import pathlib

def attendance(name):
    today = datetime.date.today()
    file = pathlib.Path('Attendance/attendance_'+ str(today) +'.csv')
    if file.exists() == False:
        f = open('Attendance/attendance_'+ str(today) +'.csv', 'w')
        writer = csv.writer(f)
        f.writelines(f"Name,Time")
        f.close()
    
    with open('Attendance/attendance_'+ str(today) +'.csv', "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtstring}")