
from roboflow import Roboflow
rf = Roboflow(api_key="dah12WWwsvZHB36zP0kM")
project = rf.workspace("findluggage").project("find-luggage")
dataset = project.version(1).download("yolov8")

