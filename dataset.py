from roboflow import Roboflow


rf = Roboflow(api_key="ZRhugjRon1Sw5gJ5Ma2g")
project = rf.workspace("ss-hwnzd").project("sign_recognition")
version = project.version(2)
dataset = version.download("yolov11")