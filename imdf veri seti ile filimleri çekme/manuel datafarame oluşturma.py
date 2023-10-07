import random

randomlist1 = random.sample(range(15,25), 2)
randomlist2 = random.sample(range(15,25), 2)

randomlistofists = [randomlist1,randomlist2]
print(randomlistofists) 

columns = ["sıcaklık 1.gün","sıcaklık 2. gün"]
mydataframe = pd.DataFrame(randomlistofists,index = ["İst","Ankara"],columns = columns)
print(mydataframe)