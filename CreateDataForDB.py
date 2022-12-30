import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
cred = credentials.Certificate("E:/Project/PBL4/serviceAccountKey.json")
firebase_admin.initialize_app(cred,
                              {'databaseURL': "https://facerecognitionrealtime-default-rtdb.firebaseio.com/"})


ref = db.reference('Users')

data = {
    "HuyenTrang":
        {
            "name": "HuyenTrang",
            "last_checked": "2022-12-18 00:54:34"
        },
    "KhanhPhuong":
        {
            "name": "KhanhPhuong",
            "last_checked": "2022-12-18 00:54:34"
        },
    "TrangDao":
        {
            "name": "TrangDao",
            "last_checked": "2022-12-18 00:54:34"
        },
    "KhanhBeo":
        {
            "name": "KhanhBeo",
            "last_checked": "2022-12-18 00:54:34"
        },
    "HoangLong":
        {
            "name": "HoangLong",
            "last_checked": "2022-12-18 00:54:34"
        },

}


for key, value in data.items():
    ref.child(key).set(value)
